# Shared Drives, OneDrive and File Access

## Where to store files

| Location | Use for | Backup |
|---|---|---|
| OneDrive | Your own work files | Continuous; 93-day version history |
| SharePoint / Teams files | Team documents | Continuous; 93-day version history |
| `\\fs01\departments` (S: drive) | Legacy department shares, being retired | Nightly, kept 30 days |
| Laptop desktop / `C:\Temp` | Nothing important | **Not backed up** |

## Requesting access to a folder

Access is granted by the folder owner, not by IT. In SharePoint, click **Request access**.
For the S: drive, open an IT portal request with the full path; the request is routed to the
owner listed for that folder.

## Restoring a deleted or overwritten file

- **OneDrive / SharePoint:** open the site's Recycle Bin (kept 93 days), or right-click a file → **Version history** to restore an earlier version.
- **S: drive:** right-click the folder → Properties → **Previous Versions** for snapshots from the last 30 days. Older restores need a ticket.

## Mapped drive missing

Mapped drives appear only when connected to the company network or VPN. Connect the VPN,
then sign out and in again. To map manually: File Explorer → This PC → **Map network
drive** → `S:` → `\\fs01\departments`.
