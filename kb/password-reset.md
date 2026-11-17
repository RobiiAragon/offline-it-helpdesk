# Password Reset and Account Lockout

Applies to: all employee accounts (single sign-on, email, VPN, Wi-Fi).

## Password policy

- Minimum length is 14 characters. Passphrases such as `river-lamp-cactus-42` are encouraged.
- Passwords expire every 180 days. You receive email reminders 14, 7 and 1 day before expiry.
- The last 12 passwords cannot be reused.
- Never share your password with anyone, including IT staff. The Service Desk will never ask for it.

## Self-service reset

1. Go to the self-service portal at `https://sspr.example.com` from any browser.
2. Enter your work email and complete the MFA prompt (Authenticator app or SMS).
3. Choose a new password that meets the policy above.
4. Wait up to 15 minutes for the new password to sync to Wi-Fi and VPN.
5. On your laptop, lock the screen (Windows+L or Ctrl+Cmd+Q) and unlock it with the new password so the cached credential updates.

## Account lockout

An account locks automatically for **30 minutes** after **10 failed sign-in attempts**.
Common causes are an old password saved on a phone (email app), a mapped network drive, or
a Wi-Fi profile on a second device.

- Wait 30 minutes, or call the Service Desk at extension 4357 to unlock immediately.
- After unlocking, update the password on every phone and tablet before signing in again.
- Repeated lockouts with no obvious cause should be reported; the Security team checks the sign-in logs for password-spraying attempts.

## Escalation

If self-service reset fails because you no longer have your MFA device, see the
"Multi-Factor Authentication" runbook for the identity-verification process.
