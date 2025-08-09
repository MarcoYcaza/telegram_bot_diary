systemctl daemon-reload
systemctl restart kathy-bot.service
journalctl -u kathy-bot.service