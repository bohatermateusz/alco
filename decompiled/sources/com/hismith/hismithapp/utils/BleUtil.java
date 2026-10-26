package com.hismith.hismithapp.utils;

import com.hismith.hismithapp.model.BleAdvertisedData;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/* loaded from: classes3.dex */
public class BleUtil {
    private static final String TAG = "BleUtil";

    public static class ParsedAd {
        int flags;
        String localName;
        short manufacturer;
        List<UUID> uuids = new ArrayList();

        public String getLocalName() {
            return this.localName;
        }
    }

    public static BleAdvertisedData parseAdertisedData(byte[] bArr) {
        byte b;
        ArrayList arrayList = new ArrayList();
        String str = null;
        if (bArr == null) {
            return new BleAdvertisedData(arrayList, null);
        }
        try {
            ByteBuffer order = ByteBuffer.wrap(bArr).order(ByteOrder.LITTLE_ENDIAN);
            while (order.remaining() > 2 && (b = order.get()) != 0) {
                byte b2 = order.get();
                if (b2 == 2 || b2 == 3) {
                    while (b >= 2) {
                        arrayList.add(UUID.fromString(String.format("%08x-0000-1000-8000-00805f9b34fb", Short.valueOf(order.getShort()))));
                        b = (byte) (b - 2);
                    }
                } else if (b2 == 6 || b2 == 7) {
                    while (b >= 16) {
                        arrayList.add(new UUID(order.getLong(), order.getLong()));
                        b = (byte) (b - 16);
                    }
                } else if (b2 != 9) {
                    int position = (order.position() + b) - 1;
                    if (position > 0 && position < order.capacity()) {
                        order.position(position);
                    }
                } else {
                    byte[] bArr2 = new byte[b - 1];
                    order.get(bArr2);
                    try {
                        str = new String(bArr2, "utf-8");
                    } catch (UnsupportedEncodingException e) {
                        e.printStackTrace();
                    }
                }
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        return new BleAdvertisedData(arrayList, str);
    }

    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    /* JADX WARN: Removed duplicated region for block: B:21:0x00b7 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:25:0x000f A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static ParsedAd parseScanRecodeData(byte[] bArr) {
        byte b;
        int i;
        ParsedAd parsedAd = new ParsedAd();
        ByteBuffer order = ByteBuffer.wrap(bArr).order(ByteOrder.LITTLE_ENDIAN);
        while (order.remaining() > 2 && (b = order.get()) != 0) {
            byte b2 = order.get();
            int i2 = (byte) (b - 1);
            if (b2 != -1) {
                if (b2 != 20) {
                    if (b2 != 21) {
                        switch (b2) {
                            case 1:
                                parsedAd.flags = order.get();
                                i = i2 - 1;
                                break;
                            case 4:
                            case 5:
                                while (i2 >= 4) {
                                    parsedAd.uuids.add(UUID.fromString(String.format("%08x-0000-1000-8000-00805f9b34fb", Integer.valueOf(order.getInt()))));
                                    i2 = (byte) (i2 - 4);
                                }
                                break;
                            case 8:
                            case 9:
                                byte[] bArr2 = new byte[i2];
                                order.get(bArr2, 0, i2);
                                parsedAd.localName = new String(bArr2).trim();
                                i2 = 0;
                                break;
                        }
                        if (i2 <= 0) {
                            order.position(order.position() + i2);
                        }
                    }
                    while (i2 >= 16) {
                        parsedAd.uuids.add(new UUID(order.getLong(), order.getLong()));
                        i2 = (byte) (i2 - 16);
                    }
                    if (i2 <= 0) {
                    }
                }
                while (i2 >= 2) {
                    parsedAd.uuids.add(UUID.fromString(String.format("%08x-0000-1000-8000-00805f9b34fb", Short.valueOf(order.getShort()))));
                    i2 = (byte) (i2 - 2);
                }
                if (i2 <= 0) {
                }
            } else {
                parsedAd.manufacturer = order.getShort();
                i = i2 - 2;
            }
            i2 = (byte) i;
            if (i2 <= 0) {
            }
        }
        return parsedAd;
    }
}
