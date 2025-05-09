#!/bin/bash
# MIDAS3 AWS EC2 Deployment Script for Amazon Linux 2023

# Script assumes it is run from the root of the MIDAS3 repository

# 1. System Setup
sudo dnf check-update -y
sudo dnf upgrade -y
echo "Installing core packages including Nginx and build tools..."
sudo dnf install -y python3 python3-pip nginx git openssl gcc python3-devel openssl-devel libffi-devel cargo firewalld

echo "Reloading systemd manager configuration after package installations..."
sudo systemctl daemon-reload

echo "Verifying Nginx service unit file..."
if ! sudo systemctl list-unit-files | grep -q '^nginx.service'; then
    echo "ERROR: nginx.service unit file not found after installation."
    echo "Please check dnf logs (e.g., /var/log/dnf.log) and ensure the nginx package installed correctly."
    exit 1
else
    echo "Nginx service unit file found."
fi

# 2. Python Environment
python3.9 -m venv venv
source venv/bin/activate
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Clearing pip cache..."
pip cache purge

echo "Checking Python and pip versions in venv..."
python --version
pip --version


echo "Cryptography installed successfully or was already satisfied."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: 'pip install -r requirements.txt' failed. See errors above."
    exit 1
fi

echo "Verifying Flask installation..."
pip show flask
if [ $? -ne 0 ]; then
    echo "ERROR: Flask is not installed after 'pip install -r requirements.txt'."
    exit 1
fi
echo "Flask installation verified."
pip install gunicorn==21.2.0 gevent==24.2.1

# 3. Configuration
cat > .env <<EOL
FLASK_ENV=production
SECRET_KEY=$(openssl rand -hex 32)
DATABASE_URL=sqlite:///$(pwd)/data/conversations.db
OLLAMA_HOST=http://localhost:11434
EOL

# 3.1. Set Permissions
echo "Setting permissions for project directory..."
PROJECT_DIR=$(pwd)
# Set for root user
sudo chown -R root:root "$PROJECT_DIR"
# Grant read+execute for user (ec2-user) on all files/dirs in the project.
sudo chmod -R u+rX "$PROJECT_DIR"
# Specifically for static assets to be served by Nginx (nginx user is 'other')
# and for general access if needed:
# Allow 'others' (e.g., nginx user) to traverse into the project directory.
sudo chmod o+x "$PROJECT_DIR"
# Allow 'others' to traverse into 'static' and read files within.
# Ensure static directory itself is executable by others.
sudo chmod o+x "$PROJECT_DIR/static"
# Ensure subdirectories within static are executable by others.
sudo find "$PROJECT_DIR/static" -type d -exec sudo chmod o+x {} \;
# Ensure files within static (and its subdirectories) are readable by others.
sudo find "$PROJECT_DIR/static" -type f -exec sudo chmod o+r {} \;


# 4. Nginx Setup
sudo bash -c 'cat > /etc/nginx/conf.d/midas.conf <<EOL
server {
    listen 80;
    server_name _;
    
    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /static/ {
        alias $(pwd)/static/;
        expires 30d; # Optional: Add caching for static assets
    }
}
EOL'

# Ensure Nginx is started and enabled correctly
sudo systemctl daemon-reload # Reload systemd manager configuration
sudo systemctl start nginx
if sudo systemctl is-active --quiet nginx; then
    echo "Nginx started successfully."
    sudo systemctl enable nginx
    echo "Nginx enabled successfully."
    sudo systemctl restart nginx # Restart to apply new midas.conf
else
    echo "ERROR: Nginx failed to start. Please check the Nginx installation and logs."
    sudo systemctl status nginx
    exit 1
fi

# 5. Systemd Service (Optional)
read -p "Do you want to set up MIDAS as a Systemd service? (y/N): " setup_service
if [[ "$setup_service" =~ ^[Yy]$ ]]; then
    # Create systemd service
    echo "Creating systemd service..."

    # Clean up old service first
    echo "Stopping and removing existing MIDAS service..."
    sudo systemctl stop midas || true
    sudo systemctl disable midas || true
    sudo rm -f /etc/systemd/system/midas.service
    sudo systemctl daemon-reload

    # Create new service
    cat <<EOF | sudo tee /etc/systemd/system/midas.service > /dev/null
[Unit]
Description=MIDAS Application
After=network.target

[Service]
User=$USER
WorkingDirectory=$PWD
Environment="PATH=$PWD/venv/bin:\$PATH"
ExecStart=$PWD/venv/bin/gunicorn --bind 0.0.0.0:5000 --timeout 1800 --workers 3 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

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
sudo firewall-cmd --permanent --zone=docker --add-service=http
sudo firewall-cmd --permanent --zone=docker --add-service=ssh # Ensure SSH is allowed
sudo firewall-cmd --permanent --zone=docker --add-port=8188/tcp # Enable port 8188 for ComfyUI
sudo firewall-cmd --reload

# 7. Data Directory
echo "Creating data directories..."
sudo mkdir -p $INSTALL_DIR/data/uploads
sudo mkdir -p $INSTALL_DIR/data/conversations
sudo mkdir -p $INSTALL_DIR/data/models

# ComfyUI output directory
echo "Creating ComfyUI output directory..."
sudo mkdir -p /MIDAS_standalone/ComfyUI/output
sudo chown -R $USER:$USER /MIDAS_standalone
sudo chmod -R 755 /MIDAS_standalone

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
