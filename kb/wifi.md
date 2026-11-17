# Wi-Fi Networks

| Network | Who | Authentication |
|---|---|---|
| `CORP-SECURE` | Company laptops and phones | Certificate (automatic) |
| `CORP-BYOD` | Personal phones | Work email + password, MFA every 30 days |
| `GUEST` | Visitors | Daily code from reception, 8-hour sessions |
| `PLANT-IOT` | Production equipment only | Managed by OT engineering; not for users |

## Company laptop will not join CORP-SECURE

1. Make sure the laptop has been on the company network (wired or VPN) in the last 90 days; the Wi-Fi certificate renews automatically only then.
2. Forget the network and reconnect.
3. Restart the laptop. On Windows, run `gpupdate /force` while connected by cable or VPN.
4. If it still fails, the certificate has probably expired. Bring the laptop to the Service Desk or connect by cable and open a ticket.

## Personal phone on CORP-BYOD

Select `CORP-BYOD`, choose **Do not validate** for the certificate if your phone asks,
and sign in with your work email. After a password change you must forget the network and
reconnect, otherwise your account may lock (see "Password Reset and Account Lockout").

## Guests

Visitors get a daily `GUEST` code from reception. Guest Wi-Fi is internet-only and isolated
from internal systems. Employees must not give their own credentials to visitors.
