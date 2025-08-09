systemctl daemon-reload
systemctl restart diary.service
journalctl -u diary.service
