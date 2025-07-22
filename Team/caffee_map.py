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

    df1, df2, df3 = [pd.read_csv('../dataFile/area_struct.csv')
                     , pd.read_csv('../dataFile/area_map.csv')
                     , pd.read_csv('../dataFile/area_category.csv')
                    ]

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df3.columns = df3.columns.str.strip()
    
    result = merged_data(df1, df2, df3)

    #print(result)
    #CSV 파일 생성
    result.to_csv('../dataFile/caffee_data.csv', index=False)
    

if __name__ == '__main__':
    main()