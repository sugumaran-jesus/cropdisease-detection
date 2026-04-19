bind = "0.0.0.0:10000"      # ✅ add this line
workers = 1
worker_class = "sync"
timeout = 300
max_requests = 1000
max_requests_jitter = 50