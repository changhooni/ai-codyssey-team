from flask import Flask

app = Flask(__name__)

def selection_sort(data):
    numbers = [float(num) for num in data.split()] # 입력한 값을 split 공백 처리를 하고 반복문으로 num에 값을 넣어줌 값을 다시 float로 변환함
    n = len(numbers) # 입력된 데이터에 갯수
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if numbers[j] < numbers[min_index]:
                min_index = j
        numbers[i], numbers[min_index] = numbers[min_index], numbers[i]
    return numbers


def main():
    # 사용자로부터 숫자 입력 받기
    input_str = input("정렬할 숫자들을 입력하세요 (공백으로 구분): ")
    #numbers = list(map(int, input_str.split()))

    # 선택 정렬 수행
    sorted_numbers = selection_sort(input_str)

    # 결과 출력
    print("정렬된 숫자:", sorted_numbers)

if __name__ == "__main__":
    main()