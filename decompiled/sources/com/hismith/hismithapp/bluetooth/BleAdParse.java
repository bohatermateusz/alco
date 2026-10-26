package com.hismith.hismithapp.bluetooth;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.UUID;

/* loaded from: classes3.dex */
public class BleAdParse {
    public static String byte2hex(byte[] bArr) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bArr) {
            String upperCase = Integer.toHexString(b & 255).toUpperCase();
            if (upperCase.length() == 1) {
                sb.append("0").append(upperCase);
            } else {
                sb.append(upperCase);
            }
        }
        return sb.toString();
    }

    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    /* JADX WARN: Removed duplicated region for block: B:23:0x00c2 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:27:0x000f A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static ParsedAd parseScanRecodeData(byte[] bArr) {
        byte b;
        ParsedAd parsedAd = new ParsedAd();
        ByteBuffer order = ByteBuffer.wrap(bArr).order(ByteOrder.LITTLE_ENDIAN);
        while (order.remaining() > 2 && (b = order.get()) != 0 && b <= order.remaining()) {
            byte b2 = order.get();
            int i = (byte) (b - 1);
            if (b2 != -1) {
                if (b2 != 20) {
                    if (b2 != 21) {
                        switch (b2) {
                            case 1:
                                parsedAd.flags = order.get();
                                i = (byte) (i - 1);
                                break;
                            case 4:
                            case 5:
                                while (i >= 4) {
                                    parsedAd.uuids.add(UUID.fromString(String.format("%08x-0000-1000-8000-00805f9b34fb", Integer.valueOf(order.getInt()))));
                                    i = (byte) (i - 4);
                                }
                                break;
                            case 8:
                            case 9:
                                byte[] bArr2 = new byte[i];
                                order.get(bArr2, 0, i);
                                parsedAd.localName = new String(bArr2).trim();
                                break;
                        }
                        if (i <= 0) {
                            order.position(Math.min(order.position() + i, order.capacity()));
                        }
                    }
                    while (i >= 16) {
                        parsedAd.uuids.add(new UUID(order.getLong(), order.getLong()));
                        i = (byte) (i - 16);
                    }
                    if (i <= 0) {
                    }
                }
                while (i >= 2) {
                    parsedAd.uuids.add(UUID.fromString(String.format("%08x-0000-1000-8000-00805f9b34fb", Short.valueOf(order.getShort()))));
                    i = (byte) (i - 2);
                }
                if (i <= 0) {
                }
            } else {
                byte[] bArr3 = new byte[i];
                order.get(bArr3, 0, i);
                parsedAd.diyInfo = byte2hex(bArr3);
            }
            i = 0;
            if (i <= 0) {
            }
        }
        return parsedAd;
    }
}
