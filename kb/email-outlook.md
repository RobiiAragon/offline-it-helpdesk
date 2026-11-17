# Email and Outlook

## Mailbox limits

- Mailbox size: 50 GB. Online archive: an additional 100 GB, enabled automatically when the mailbox reaches 90%.
- Maximum attachment size: 25 MB. For larger files, share a OneDrive link instead.
- External recipients per message: 500.

## Outlook will not open or keeps asking for a password

1. Close Outlook completely and check Task Manager for leftover `OUTLOOK.EXE` processes.
2. Start Outlook in safe mode with `outlook.exe /safe` to rule out add-ins.
3. If the password prompt loops, open **Credential Manager → Windows Credentials** and remove entries containing `MicrosoftOffice`.
4. Create a new Outlook profile: Control Panel → Mail → **Show Profiles → Add**.

## Shared mailboxes

Shared mailboxes (for example `purchasing@example.com`) appear automatically in Outlook
about one hour after access is granted. Access requests go through the IT portal and need
the mailbox owner's approval.

## Out of office

Outlook → File → **Automatic Replies**. Set a separate message for external senders. On the
web, use Settings → Mail → Automatic replies.

## Recovering deleted email

Items deleted from Deleted Items can be restored for **30 days**: in Outlook, right-click
Deleted Items → **Recover Deleted Items**. After 30 days, the Service Desk can request a
restore from backup only for legal or audit needs, approved by your director.
