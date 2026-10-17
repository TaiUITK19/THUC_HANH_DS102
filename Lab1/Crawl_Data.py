import requests
import pandas as pd


categories = [17, 27, 20, 32, 31, 16, 26, 23, 22, 33]
paper_list = []
for cat in categories:
    url = f"https://api.openalex.org/works?filter=primary_topic.field.id:{cat}&per-page=200"


    response = requests.get(url)

    if response.status_code == 200:
        papers = response.json().get("results", [])

        for paper in papers:

            ten_tac_gia = []
            for a in paper.get("authorships", []):
                ten_tac_gia.append(a["author"]["display_name"])

            topic = paper.get("primary_topic")
            paper_list.append({
                "Mã bài báo": paper.get("id"),
                "Tiêu đề": paper.get("title"),
                "Năm xuất bản": paper.get("publication_year"),
                "DOI": paper.get("doi"),
                "Loại tài liệu": paper.get("type"),
                "Số lượt trích dẫn": paper.get("cited_by_count"),
                "Chủ đề": topic["display_name"],
                "Chuyên ngành": topic["subfield"]["display_name"],
                "Tác giả": "; " .join(ten_tac_gia),
                "Số tác giả": len(ten_tac_gia),
                "Nhãn": topic["field"]["display_name"],
            })
        df = pd.DataFrame(paper_list)
        df.to_csv("openalex_papers.csv", index=False, encoding="utf-8-sig")
    else:
        print(f"Không thể truy cập dữ liệu. Mã trạng thái: {response.status_code}")

print("Dữ liệu đã được lưu vào openalex_papers.csv")