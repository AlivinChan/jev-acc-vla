"""Exploratory tau=0 ablation; reuse the exact frozen Laya service and prompt."""
import laya_service

laya_service.MINIMUM_PROBABILITY = 0.0

if __name__ == '__main__':
    raise SystemExit(laya_service.main())
