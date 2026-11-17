# VPN Remote Access

The company uses GlobalProtect VPN. Portal address: `vpn.example.com`.

## Installing

- Company laptops have the client pre-installed. If it is missing, install **GlobalProtect** from Software Center (Windows) or Self Service (macOS).
- Personal devices are not allowed on the VPN.

## Connecting

1. Open GlobalProtect from the system tray (Windows) or menu bar (macOS).
2. Enter `vpn.example.com` as the portal if prompted.
3. Sign in with your work email and approve the MFA request.
4. The status changes to **Connected**. Internal sites such as `https://intranet.example.com` now load.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Portal not reachable" | Check internet access first. Hotel and airport Wi-Fi often need a captive-portal login in a browser. |
| Connects then drops every few minutes | Switch from Wi-Fi to a wired connection or phone hotspot; disable any other VPN software. |
| Authentication failed | Your password may have expired. Reset it at `https://sspr.example.com` and wait 15 minutes. |
| Connected but internal sites do not load | Disconnect, run `ipconfig /flushdns` (Windows) or `sudo dscacheutil -flushcache` (macOS), reconnect. |
| Very slow file transfers | Large file copies over VPN are throttled; use OneDrive or the file-transfer portal instead. |

Collect logs before calling the Service Desk: GlobalProtect → Settings → Troubleshooting →
**Collect Logs**, and attach the zip file to your ticket.
