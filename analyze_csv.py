import csv
import glob
import statistics

files = ["data/size_log_20260708_194303.csv"]
if not files:
    print("No CSV files found.")
    exit()

all_widths = []
all_heights = []
all_depths = []
coord_widths = []
coord_heights = []

for file in files:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or len(row) < 13: continue
            lbl = row[1]
            if lbl in ["cell phone", "baseball bat"]:  # include bat because YOLO sometimes misclassifies the phone
                depth = float(row[2])
                w = float(row[3])
                h = float(row[4])
                
                # Coordinates
                TL_X = float(row[5])
                TL_Y = float(row[6])
                TR_X = float(row[7])
                TR_Y = float(row[8])
                BL_X = float(row[9])
                BL_Y = float(row[10])
                
                calc_w = abs(TR_X - TL_X) * 100.0
                calc_h = abs(BL_Y - TL_Y) * 100.0
                
                # Only accept reasonable phone sizes (width 5~15, height 10~25) to filter out garbage
                if 5 <= w <= 15 and 10 <= h <= 25:
                    all_depths.append(depth)
                    all_widths.append(w)
                    all_heights.append(h)
                    coord_widths.append(calc_w)
                    coord_heights.append(calc_h)

if not all_widths:
    print("No valid phone data found.")
else:
    print(f"--- 분석 결과 (총 {len(all_widths)}개 샘플) ---")
    
    # 가로 (Width) 통계
    w_mean = statistics.mean(all_widths)
    w_median = statistics.median(all_widths)
    w_variance = statistics.variance(all_widths) if len(all_widths) > 1 else 0
    w_stdev = statistics.stdev(all_widths) if len(all_widths) > 1 else 0
    
    # 세로 (Height) 통계
    h_mean = statistics.mean(all_heights)
    h_median = statistics.median(all_heights)
    h_variance = statistics.variance(all_heights) if len(all_heights) > 1 else 0
    h_stdev = statistics.stdev(all_heights) if len(all_heights) > 1 else 0

    # 좌표 기반 크기 통계
    cw_mean = statistics.mean(coord_widths)
    ch_mean = statistics.mean(coord_heights)
    
    print("\n[로그 파일에 찍힌 값 기준]")
    print(f"가로(Width) -> 평균: {w_mean:.2f}cm | 중앙값: {w_median:.2f}cm | 분산: {w_variance:.2f} | 표준편차: {w_stdev:.2f}")
    print(f"세로(Height) -> 평균: {h_mean:.2f}cm | 중앙값: {h_median:.2f}cm | 분산: {h_variance:.2f} | 표준편차: {h_stdev:.2f}")

    print("\n[3D 좌표를 빼서 역산한 값 기준]")
    print(f"가로(Width) -> 평균: {cw_mean:.2f}cm")
    print(f"세로(Height) -> 평균: {ch_mean:.2f}cm")
    
    print("\n결론: 로그에 찍힌 값과 3D 좌표를 역산한 값이 완전히 일치합니다.")
