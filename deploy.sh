#!/bin/bash
# MIDAS3 AWS EC2 Deployment Script

# Script assumes it is run from the root of the MIDAS3 repository

# 1. System Setup
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv nginx git

# 2. Python Environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn==21.2.0 gevent==24.2.1

# 3. Configuration
# 4. Configuration
cat > .env <<EOL
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=sqlite:///$(pwd)/data/conversations.db
OLLAMA_HOST=http://localhost:11434
EOL

# 5. Nginx Setup
sudo bash -c 'cat > /etc/nginx/sites-available/midas <<EOL
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        include proxy_params;
    }

    location /static/ {
        alias $(pwd)/static/;
    }
}
EOL'

sudo ln -sf /etc/nginx/sites-available/midas /etc/nginx/sites-enabled
sudo nginx -t && sudo systemctl restart nginx

# 6. Systemd Service (Optional)
read -p "Do you want to set up MIDAS as a Systemd service? (y/N): " setup_service
if [[ "$setup_service" =~ ^[Yy]$ ]]; then
    sudo bash -c 'cat > /etc/systemd/system/midas.service <<EOL
[Unit]
Description=MIDAS3 Application
After=network.target

[Service]
User=$USER
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

# 7. Security
sudo ufw allow 80
sudo ufw allow 22
sudo ufw --force enable

# 8. Data Directory
mkdir -p data

# 9. Initial Setup Complete
echo "╔══════════════════════════════════════════════╗"
echo "║          MIDAS3 Deployment Complete          ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ Access: http://$(curl -s ifconfig.me)        ║"
echo "║                                              ║"
echo "║ Next Steps:                                  ║"
echo "║ 1. Setup DNS (Route53 or other)              ║"
echo "║ 2. Configure HTTPS with Let's Encrypt        ║"
echo "║ 3. Monitor logs: journalctl -u midas -f      ║"
echo "╚══════════════════════════════════════════════╝"
