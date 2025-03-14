import json
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import APIRouter, HTTPException
import io
import base64
from app.api.divergence import analyze  # analyze 함수 가져오기

router = APIRouter()

@router.get("/test/divergence")
async def get_stock_graph():
    # JSON 파일에서 데이터 로드
    try:
        with open('app/data/rdfn_history.json', 'r') as file:
            data = json.load(file)
        
        df = pd.DataFrame(data['history'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print(df)
        # trend_score 계산
        trend_scores = []
        for index, row in df.iterrows():
            print(index, row)
            single_day_df = pd.DataFrame([row])
            print('single_day_df', '*****', single_day_df) 
            analysis_result = analyze(single_day_df)
            print(index, analysis_result)
            trend_scores.append(float(analysis_result['trend_score']))  # trend_score 추가


        # 그래프 생성
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Close'], label='Closing Prices', color='blue')
        plt.plot(df.index, trend_scores, label='Trend Score', color='red')  # trend_score 그래프 추가
        plt.title('Stock Closing Prices and Trend Scores Over the Last Year')
        plt.xlabel('Date')
        plt.ylabel('Price / Trend Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        # 그래프를 BytesIO 객체에 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # 이미지를 base64로 인코딩
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {
            "image": f"data:image/png;base64,{image_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")