import requests
import pandas as pd


categories = ["software-dev", "customer-support", "design", "marketing", "sales", "product", "business", "data", "finance", "hr", "legal", "operations", "qa", "teaching"]
job_list = []
for cat in categories:
    url = f"https://remotive.com/api/remote-jobs?category={cat}"


    response = requests.get(url)

    if response.status_code == 200:
        jobs = response.json().get("jobs", [])

        for job in jobs:
            job_list.append({
                "Mã công viẹc": job.get("id"),
                "Tên công việc": job.get("title"),
                "Công ty": job.get("company_name"),
                "Danh mục": job.get("category"),
                "Mức lương": job.get("salary") if job.get("salary") else "Thỏa thuận",
                "Loại hình": job.get("job_type"),
                "Ngày đăng": job.get("publication_date"),
                "Link chi tiết": job.get("url")
            })
        df = pd.DataFrame(job_list)
        df.to_csv("remote_jobs.csv", index=False, encoding="utf-8-sig")
    else:
        print(f"Không thể truy cập dữ liệu. Mã trạng thái: {response.status_code}")

print("Dữ liệu đã được lưu vào remote_jobs.csv")