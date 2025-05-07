#!/bin/bash
# MIDAS3 AWS EC2 Deployment Script for Amazon Linux 2023

# Script assumes it is run from the root of the MIDAS3 repository

# 1. System Setup
sudo dnf check-update -y
sudo dnf upgrade -y
sudo dnf install -y python3 python3-pip python3-venv nginx git openssl

# 2. Python Environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn==21.2.0 gevent==24.2.1

# 3. Configuration
cat > .env <<EOL
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=sqlite:///$(pwd)/data/conversations.db
OLLAMA_HOST=http://localhost:11434
EOL

# 4. Nginx Setup
sudo bash -c 'cat > /etc/nginx/conf.d/midas.conf <<EOL
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias $(pwd)/static/;
        expires 30d; # Optional: Add caching for static assets
    }
}
EOL'

sudo systemctl restart nginx
sudo systemctl enable nginx

# 5. Systemd Service (Optional)
read -p "Do you want to set up MIDAS as a Systemd service? (y/N): " setup_service
if [[ "$setup_service" =~ ^[Yy]$ ]]; then
    sudo bash -c 'cat > /etc/systemd/system/midas.service <<EOL
[Unit]
Description=MIDAS3 Application
After=network.target nginx.service

[Service]
User=ec2-user
WorkingDirectory=$(pwd)
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/venv/bin/gunicorn -w 4 -k gevent -b 127.0.0.1:5000 --access-logfile - app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOL'

    sudo systemctl daemon-reload
    sudo systemctl start midas
    sudo systemctl enable midas
    echo "MIDAS Systemd service created and started."
else
    echo "Skipping Systemd service setup. You can run MIDAS manually using:"
    echo "source venv/bin/activate && gunicorn -w 4 -k gevent -b 127.0.0.1:5000 --access-logfile - app:app"
fi

# 6. Firewall Configuration (firewalld)
sudo systemctl start firewalld
sudo systemctl enable firewalld
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=ssh # Ensure SSH is allowed
sudo firewall-cmd --reload

# 7. Data Directory
mkdir -p data

# 8. Initial Setup Complete
echo "╔══════════════════════════════════════════════╗"
echo "║ MIDAS3 Deployment Complete (Amazon Linux 2023) ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ Access: http://$(curl -s -4 ifconfig.me)     ║"
echo "║                                              ║"
echo "║ Next Steps:                                  ║"
echo "║ 1. Setup DNS (Route53 or other)              ║"
echo "║ 2. Configure HTTPS with Let's Encrypt        ║"
echo "║ (Certbot for Nginx on Amazon Linux)        ║"
echo "║ 3. Monitor logs: journalctl -u midas -f      ║"
echo "╚══════════════════════════════════════════════╝"

