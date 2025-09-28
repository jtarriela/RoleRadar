import subprocess

# url = "https://careers.bankofamerica.com/sitemap_index.xml"
    # https://careers.bankofamerica.com/en-us/job-detail/ is the job prefix

# url = "https://www.capitalonecareers.com/sitemap.xml"
    # https://www.capitalonecareers.com/job is the job prefix

url = "https://higher.gs.com" # needs a crawler 

#amazon career needs crawler


url = "https://bloomberg.avature.net/sitemap.xml"
# bloomberg needs parser 
    # funny on their https://www.bloomberg.com/robots.txt they say 
    # "# If you can read this then you should apply here https://www.bloomberg.com/careers/" 
# url = "https://www.morganstanley.com/sitemap.xml"



# url="https://careers.ansys.com/sitemap.xml" # valid but cannot parse 



# url = "https://explore.jobs.netflix.net/careers/sitemap.xml"
# url="https://www.metacareers.com/jobs/sitemap.xml" 

# microsoft -- may need to paginate https://jobs.careers.microsoft.com/global/en/job/1877695/Hardware-Engineer
# amazon needs crawler 
url = "https://careers.google.com/jobs/sitemap"
url = "https://www.wellsfargojobs.com/sitemap.xml"
url = "https://jobs.apple.com/sitemap/sitemap-jobs-en-us.xml"
outdir = "/home/jd/proj/RoleRadar/test"

subprocess.run(
    [
        "python3",
        "/home/jd/proj/sitemap-extract/sitemap_extract.py",
        "--url", url,
        "--save-dir", outdir,
    ],
    check=True
)