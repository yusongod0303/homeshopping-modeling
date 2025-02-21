import bz2

# 압축 해제
input_file = "kowiki-latest-pages-articles.xml.bz2"
output_file = "kowiki-latest-pages-articles.xml"

with bz2.BZ2File(input_file, "rb") as compressed_file:
    with open(output_file, "wb") as decompressed_file:
        decompressed_file.write(compressed_file.read())

print(f"압축 해제 완료: {output_file}")


def preprocess_wikipedia_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            # 공백 제거 및 간단한 전처리
            clean_line = line.strip()
            if clean_line:  # 비어 있는 줄 제외
                outfile.write(clean_line + "\n")
    print(f"전처리 완료: {output_file}")

# 전처리 실행
input_text = "kowiki-latest-pages-articles.xml"
preprocessed_text = "wikipedia_preprocessed.txt"
preprocess_wikipedia_data(input_text, preprocessed_text)


from gensim.models import FastText

# 학습 데이터 준비
def read_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.split()  # 단어로 분리된 리스트 반환

# 학습 데이터 경로
data_path = "wikipedia_preprocessed.txt"

# FastText 모델 학습
print("FastText 모델 학습 시작...")
sentences = list(read_sentences(data_path))
model = FastText(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4, sg=1)

# 모델 저장
model.save("fasttext_wikipedia.model")
print("FastText 모델 학습 및 저장 완료!")
