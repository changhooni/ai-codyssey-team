import pandas as pd

#df_marged = pd.merge(df1, df2, on='x')
#df_marged = pd.concat([df1, df2, df3], ignore_index=True)
def merged_data(x, y, k):
    merged_df = pd.merge(x, k
                        , on=['category']
                        , how='outer')

    merged_df = pd.merge(merged_df, y
                        , on=['x','y']
                        , how='inner')

    #정렬 순서 지정
    merged_df = merged_df.sort_values(by=['area', 'x', 'y']
                                    , ascending=True, ignore_index=True)

    #컬럼 area값이 1만 불러오기
    merged_df = merged_df[merged_df['area'] == 1]

    #NaN 값을 공백으로 처리
    merged_df = merged_df.fillna('')

    return merged_df
    #merged_df = pd.merge(merged_df, df3, on='category', how='outer')
    #df_marged = pd.merge(left=df1, right=df2, how='inner', on='x')
    #df_marged = df_marged('category').sum()

    #print(df_marged.fillna(0))
    #print(merged_df)
    #print(df3.fillna(0))

    #CSV 파일 생성
    #merged_df.to_csv('../dataFile/map_data.csv', index=False)


def main():
    #df1 = pd.read_csv('../dataFile/area_struct.csv')
    #df2 = pd.read_csv('../dataFile/area_map.csv')
    #df3 = pd.read_csv('../dataFile/area_category.csv')

    df1, df2, df3, df4 = [pd.read_csv('../dataFile/area_struct.csv')
                     , pd.read_csv('../dataFile/area_map.csv')
                     , pd.read_csv('../dataFile/area_category.csv')
                     , pd.read_csv('../dataFile/map_data.csv')
                    ]

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df3.columns = df3.columns.str.strip()
    df4.columns = df4.columns.str.strip()
    
    result = merged_data(df1, df2, df3)

    # 공사 여부가 없는 경우 0으로 대체
    df4['ConstructionSite'] = df4['ConstructionSite'].fillna(0).astype(int)

    # 구조물 종류 이름 정의
    def get_category_name(cat):
        if cat == 1 or cat == 2:
            return 'Apartment/Building'
        elif cat == 3:
            return 'MyHome'
        elif cat == 4:
            return 'BandalgomCoffee'
        else:
            return 'ConstructionSite'

    df4['category_name'] = df4['category'].apply(get_category_name)

    # 그룹별 요약 통계
    summary = df4.groupby('category_name').agg(
        전체개수=('category', 'count'),
        공사중=('ConstructionSite', 'sum')
    ).reset_index()

    summary['정상'] = summary['전체개수'] - summary['공사중']

    # 총합 행 추가
    total = pd.DataFrame([{
        'category_name': '총합',
        '전체개수': summary['전체개수'].sum(),
        '공사중': summary['공사중'].sum(),
        '정상': summary['정상'].sum()
    }])

    summary = pd.concat([summary, total], ignore_index=True)

    # 출력
    print("📊 구조물 종류별 요약 통계 리포트")
    print("────────────────────────────")
    print(f"{'종류':<20} | {'개수':^4} | {'공사 중':^6} | {'정상':^4}")
    print("────────────────────────────")
    for _, row in summary.iterrows():
        print(f"{row['category_name']:<20} | {row['전체개수']:>4} | {row['공사중']:>6} | {row['정상']:>4}")

    #print(result)
    #CSV 파일 생성
    #result.to_csv('../dataFile/caffee_data.csv', index=False)
    

if __name__ == '__main__':
    main()