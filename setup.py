import subprocess

if __name__ == "__main__":
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    subprocess.run(["kaggle", "datasets", "download", "-d", "nelgiriyewithana/emotions", "-p", "data"])