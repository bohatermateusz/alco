# HiSmith S2 / Servo 2.0 – BLE depth (stroke) protocol

Source: static analysis (JADX decompile) of `com.hismith.hismithapp` 2.7.7 (versionCode 96).
Nothing was run on a device; the byte layouts below come from the decompiled code.
Class references are under `com/hismith/hismithapp/`.

## GATT

| Purpose | Service | Characteristic | Notes |
|---|---|---|---|
| Commands (write) | `0000ffe5-0000-1000-8000-00805f9b34fb` | `0000ffe9-...` | `Constans.UUID_BLUETOOTH_DATA` / `_DATA_WRITH` (`product/ProductBaseControl.sendCommand`) |
| Notifications | `0000ffe0-...` | `0000ffe4-...` | CCCD `00002902` enabled in `BleProduct.setBleNotification` |
| Model code (read) | `0000ff90-...` | `0000ff96-...` | product id, e.g. `1101` Servo (per Buttplug device config) |
| Rename / password / OTA | ff90/ff91, ffc0/ffc1, fd00-fd02 | | not needed for control |

App requests MTU 256 and `CONNECTION_PRIORITY_HIGH`. Scan: the advertisement contains ASCII `HISMITH`
(`4849534D495448`); the 2 bytes before it are the product code.

## Frame format ("CC" family, used by Servo / S2)

```
CC <opcode> <value> <checksum>        checksum = (opcode + value) & 0xFF
```

(`ProductBaseControl.writeCCData`). The older "AA" family uses the same layout with a different header
(`AA 04 <speed> ...`); which family a device uses is the `command` field of the product record fetched
from `https://www.hismiths.com/api/` (`Factory.createProduct`: `"CC"` -> `ProductCCControl`, otherwise
`ProductAAControl`). Buttplug's driver lists `HISMITH S2` under the CC family.

## Opcodes (`product/ProductCCControl.java`)

| Opcode | Function | Value |
|---|---|---|
| `00` | get battery | 0 |
| `01` | get status; also sub-commands (below) | |
| `02` | shutdown / stop (clears queued commands) | 0 |
| `03` | speed | 0..100 |
| `04` | speed mode | |
| `05` | vibration | |
| `06` | vibration mode | |
| **`07`** | **start position (stroke start / inner limit)** | **0..100** |
| **`08`** | **stop position (stroke end / outer limit)** | **0..100** |
| `09` | smoothness | small integer (UI slider step = 1) |
| **`0A`** | **direct position (move to this depth now)** | **0..100** |
| `0B` | "piu" (thrust burst) on/off | `01` / `00` |

Opcode `01` sub-values: `A0` auto mode, `A1` manual mode, `B0`/`B1` voice off/on, `C0`/`C1` reversal off/on.

## Depth / stroke commands

Depth is expressed as a percentage 0..100 of the machine's travel.

* **Stroke window** (what the app labels "Max stroke" / "Depth" in the play-mode dialog):
  * Set start: `CC 07 <start> <07+start>`
  * Set end:   `CC 08 <end>   <08+end>`
  * Constraint enforced by the UI: `end - start >= 10` (`Constans.start_stop_position_diff = 10`); defaults start 0, end 100.
  * Examples: start 20 -> `CC 07 14 1B`; end 80 -> `CC 08 50 58`.
* **Direct position** (device type "SMS", mode type 1 – the app routes the slider to this instead of speed):
  * `CC 0A <pos> <0A+pos>`, e.g. pos 50 -> `CC 0A 32 3C`.
* **Smoothness**: `CC 09 <n> <09+n>`.
* **Speed** (reference): `CC 03 <speed> <03+speed>`, e.g. 50 -> `CC 03 32 35`; stop = `CC 03 00 03`; hard stop `CC 02 00 02`.

Ordering/timing observed in the app: commands are queued per opcode (a newer value for the same opcode
replaces a pending one) and sent one per 50 ms for CC devices (100 ms for AA). When moving the window,
the app delays the start/stop writes 15 ms / 30 ms (order swapped depending on direction) so the window
never momentarily inverts (`dialog/PreviewDialog.onPositionRangeChange`).
Writes are plain `writeCharacteristic` on `ffe9`; the app does not use a special write type.

## Notifications (ffe4)

The device answers/echoes 4-byte frames with header `CC` or `CA`, same layout:
`07` -> start position, `08` -> stop position, `09` -> smoothness, `0A` -> current position,
`00` battery, `03` speed, `01` mode flags. The app ingests these into `BleDeviceModel`
(`BleProduct$2.onCharacteristicChanged`). Frames may be split across notifications; the app buffers
a trailing partial frame. On connect the app sends `CC 01 00 01` (get status), waits 1 s, then `CC 00 00 00` (get battery).

## Unverified

* Which exact value range the S2 firmware accepts beyond 0..100 and whether `0A` is honoured on every S2 variant:
  the code only shows the app-side range.
* Whether the S2's product record says `CC` (inferred from Buttplug's `hismith-mini` family, which uses `CC` frames); the record is served by the HiSmith web API, not in the APK.
* Smoothness maximum (slider max is in layout resources, not decompiled).
