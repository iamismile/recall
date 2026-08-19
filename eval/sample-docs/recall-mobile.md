# Recall Mobile (sample eval document)

Recall ships a companion mobile app for iOS and Android.

The mobile app mirrors the desktop index over the local network. When your phone and computer are on the same Wi-Fi, the app connects to the desktop process and replicates the indexed chunks so you can search your documents from your phone without uploading anything to the cloud.

Mobile sync is protected by end-to-end encryption with a per-device key. The desktop generates a pairing code that you enter on the phone; from then on all synced content is encrypted with a key that only exists on the two devices, so even the desktop process cannot read the phone's copy in plaintext when it is at rest.

The mobile app supports the same citation experience as the desktop: answers stream in, and tapping a citation jumps to the source chunk. Voice input is also available, letting you ask questions hands-free.

Battery usage is minimized by syncing only on Wi-Fi and pausing replication when the app is backgrounded. You can force a full re-sync from the settings menu if a device falls behind.
