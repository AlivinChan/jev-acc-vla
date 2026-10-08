"""Readable Chinese PDF from the durable report; source data remain in the adjacent repository."""
from pathlib import Path
from datetime import datetime, timezone
from xml.sax.saxutils import escape
import hashlib
import json
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'.pdf_dependencies'))
from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf'
WIDTH = A4[0]-88
SOURCE = ROOT/'REPORT.md'
FONT = Path('C:/Windows/Fonts/msyh.ttc')
FONT_BOLD = Path('C:/Windows/Fonts/msyhbd.ttc')
INK = '#243B53'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    return re.sub('[\u2010-\u2015]', '-', value)


def inline(children):
    chunks = []
    links = []
    for item in children or []:
        kind = item.type
        if kind == 'text':
            chunks.append(escape(clean(item.content)))
        elif kind in ['softbreak', 'hardbreak']:
            chunks.append('<br/>' if kind == 'hardbreak' else ' ')
        elif kind == 'strong_open':
            chunks.append('<b>')
        elif kind == 'strong_close':
            chunks.append('</b>')
        elif kind == 'em_open':
            chunks.append('<i>')
        elif kind == 'em_close':
            chunks.append('</i>')
        elif kind == 'code_inline':
            chunks.append('<font color="#35536B">'+escape(clean(item.content))+'</font>')
        elif kind == 'link_open':
            href = item.attrGet('href')
            if href.startswith(('https://','http://')):
                chunks.append('<link href="'+escape(href, {'"':'&quot;'})+'" color="#176B87">')
                links.append('</link>')
            else:
                chunks.append('<font color="#176B87">')
                links.append('</font>')
        elif kind == 'link_close':
            chunks.append(links.pop())
        elif kind == 'image':
            continue
        else:
            chunks.append(escape(clean(item.content)))
    assert not links
    return ''.join(chunks)


class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ['Title', 'H1', 'H2']:
            level = {'Title':0, 'H1':0, 'H2':1}[flowable.style.name]
            self.bookmark_index = getattr(self, 'bookmark_index', 0)+1
            key = 'section_'+str(self.bookmark_index)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level, closed=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont('CJK', str(FONT), subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJKB', str(FONT_BOLD), subfontIndex=0))
    pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJKB', italic='CJK', boldItalic='CJKB')
    body = ParagraphStyle('Body', fontName='CJK', fontSize=9.5, leading=15.8, textColor=colors.HexColor(INK),
                          wordWrap='CJK', spaceAfter=8, alignment=TA_LEFT, allowWidows=0, allowOrphans=0)
    styles = {
        'body':body,
        'title':ParagraphStyle('Title', parent=body, fontName='CJKB', fontSize=21, leading=29, spaceAfter=16, keepWithNext=True),
        'h1':ParagraphStyle('H1', parent=body, fontName='CJKB', fontSize=14, leading=20, spaceBefore=13, spaceAfter=9, keepWithNext=True),
        'h2':ParagraphStyle('H2', parent=body, fontName='CJKB', fontSize=11.2, leading=17, spaceBefore=10, spaceAfter=7, keepWithNext=True),
        'cell':ParagraphStyle('Cell', parent=body, fontSize=8, leading=11.8, spaceAfter=0),
        'head':ParagraphStyle('Head', parent=body, fontName='CJKB', fontSize=8, leading=11.8, spaceAfter=0, textColor=colors.white),
        'caption':ParagraphStyle('Caption', parent=body, fontSize=8, leading=11, spaceBefore=5, spaceAfter=11, textColor=colors.HexColor('#617487')),
        'list':ParagraphStyle('List', parent=body, leftIndent=12, firstLineIndent=0, bulletIndent=0, spaceAfter=6),
        'code':ParagraphStyle('Code', parent=body, fontSize=8.2, leading=12, backColor=colors.HexColor('#F2F5F8'), borderPadding=7),
    }
    parser = MarkdownIt('commonmark').enable('table')
    tokens = parser.parse(SOURCE.read_text(encoding='utf-8'))
    story = []
    images = {}
    table_count = 0
    lists = []
    pending_bullet = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == 'heading_open':
            level = int(token.tag[1:])
            style = styles['title' if level == 1 else 'h1' if level == 2 else 'h2']
            story.append(Paragraph(inline(tokens[index+1].children), style))
            index += 3
            continue
        if token.type == 'paragraph_open':
            children = tokens[index+1].children or []
            for child in children:
                if child.type != 'image':
                    continue
                path = (ROOT/child.attrGet('src')).resolve()
                assert path.is_relative_to(ROOT) and path.is_file()
                images[path.relative_to(ROOT).as_posix()] = sha(path)
                w,h = ImageReader(str(path)).getSize()
                scale = min(WIDTH/w, 380/h)
                picture = Image(str(path), width=w*scale, height=h*scale)
                picture.hAlign = 'LEFT'
                story.append(KeepTogether([Spacer(1,7),picture,Paragraph(escape(clean(child.content)),styles['caption'])]))
            text = inline(children)
            if text.strip():
                story.append(Paragraph(text, styles['list'] if lists else body, bulletText=pending_bullet))
                pending_bullet = None
            index += 3
            continue
        if token.type == 'table_open':
            rows = []
            current = []
            in_head = False
            index += 1
            while tokens[index].type != 'table_close':
                item = tokens[index]
                if item.type == 'thead_open':
                    in_head = True
                elif item.type == 'thead_close':
                    in_head = False
                elif item.type == 'tr_open':
                    current = []
                elif item.type == 'tr_close':
                    rows.append(current)
                elif item.type in ['th_open','td_open']:
                    cell_text = inline(tokens[index+1].children)
                    if in_head:
                        cell_text = cell_text.replace('；Smol70','；<br/>Smol70').replace(' Smol70','<br/>Smol70')
                    current.append(Paragraph(cell_text,styles['head' if in_head else 'cell']))
                index += 1
            n = len(rows[0])
            assert rows and all(len(row) == n for row in rows)
            weights = []
            for column in range(n):
                text = rows[0][column].getPlainText()
                sizes = sorted(len(row[column].getPlainText()) for row in rows)
                typical = sizes[len(sizes)//2]
                weight = min(2.2,max(.85,max(len(text)*.75,typical)/8))
                if column == 0:
                    weight = max(weight,1.55 if n>=5 else 1.15)
                if text == 'H':
                    weight = .45
                weights.append(weight)
            widths = [WIDTH*w/sum(weights) for w in weights]
            fixed = {c:34 for c in range(n) if rows[0][c].getPlainText() == 'H'}
            if fixed:
                available = WIDTH-sum(fixed.values())
                total_weight = sum(w for c,w in enumerate(weights) if c not in fixed)
                widths = [fixed[c] if c in fixed else available*w/total_weight for c,w in enumerate(weights)]
            table = Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT',spaceBefore=5,spaceAfter=12)
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor(INK)),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F8')]),
                ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#BAC8D6')),
                ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#D9E2EC')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ]))
            story.append(table)
            table_count += 1
        elif token.type in ['bullet_list_open','ordered_list_open']:
            lists.append(dict(ordered=token.type=='ordered_list_open',next=int(token.attrGet('start') or 1)))
        elif token.type == 'list_item_open':
            pending_bullet = str(lists[-1]['next'])+'.' if lists[-1]['ordered'] else '•'
            lists[-1]['next'] += 1
        elif token.type in ['bullet_list_close','ordered_list_close']:
            lists.pop()
            pending_bullet = None
        elif token.type in ['fence','code_block']:
            story.append(Paragraph(escape(clean(token.content)).replace('\n','<br/>'),styles['code']))
        elif token.type == 'hr':
            story.append(HRFlowable(width='100%',thickness=.5,color=colors.HexColor('#D9E2EC'),spaceBefore=8,spaceAfter=12))
        index += 1
    report_sha = sha(SOURCE)
    output = OUT/'LAYA阈值诊断与长动作块优化报告.pdf'
    doc = ReportDoc(str(output),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=49,bottomMargin=43,
                    title='LAYA阈值诊断与长动作块优化补充报告',author='实验记录',subject='冻结SmolVLA、二项LAYA门控及对照实验')

    def page(canvas, document):
        canvas.saveState()
        canvas.setFont('CJK',7.8)
        canvas.setFillColor(colors.HexColor('#708497'))
        canvas.drawString(44,A4[1]-28,'LAYA / 冻结 VLA 长动作块实验')
        canvas.drawRightString(A4[0]-44,A4[1]-28,'独立实验报告')
        canvas.setStrokeColor(colors.HexColor('#D9E2EC'))
        canvas.setLineWidth(.4)
        canvas.line(44,32,A4[0]-44,32)
        canvas.drawString(44,20,'来源：REPORT.md / SHA256 '+report_sha[:12])
        canvas.drawRightString(A4[0]-44,20,'第 '+str(document.page)+' 页')
        canvas.restoreState()

    doc.build(story,onFirstPage=page,onLaterPages=page)
    reader = PdfReader(str(output))
    text = '\n'.join(p.extract_text() or '' for p in reader.pages)
    assert all(len(p.extract_text() or '') > 30 for p in reader.pages)
    assert '阈值' in text and 'Smol70' in text and 'VLASH' in text and '30%' in text
    assert '\ufffd' not in text
    assert '**' not in text, 'Unrendered Markdown emphasis must be corrected'
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(),pdf=str(output),pdf_sha256=sha(output),
        report_sha256=report_sha,source_figure_sha256=images,tables=table_count,pages=len(reader.pages),
        extracted_characters=len(text),visual_review_status='render_and_inspect_required',
        generating_script_sha256=sha(Path(__file__)),font_files=[str(FONT),str(FONT_BOLD)],
        interpretation='Markdown is the content source. Local evidence links are rendered as readable labels; open the adjacent Markdown for repository navigation.')
    (ROOT/'checks/pdf_build.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    main()
