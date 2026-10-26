package com.hismith.hismithapp.utils;

import android.os.ParcelUuid;
import android.util.SparseArray;
import androidx.collection.ArrayMap;
import com.google.common.base.Ascii;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/* loaded from: classes2.dex */
public class ScanRecordUtil {
    private static final int DATA_TYPE_FLAGS = 1;
    private static final int DATA_TYPE_LOCAL_NAME_COMPLETE = 9;
    private static final int DATA_TYPE_LOCAL_NAME_SHORT = 8;
    private static final int DATA_TYPE_MANUFACTURER_SPECIFIC_DATA = 255;
    private static final int DATA_TYPE_SERVICE_DATA = 22;
    private static final int DATA_TYPE_SERVICE_UUIDS_128_BIT_COMPLETE = 7;
    private static final int DATA_TYPE_SERVICE_UUIDS_128_BIT_PARTIAL = 6;
    private static final int DATA_TYPE_SERVICE_UUIDS_16_BIT_COMPLETE = 3;
    private static final int DATA_TYPE_SERVICE_UUIDS_16_BIT_PARTIAL = 2;
    private static final int DATA_TYPE_SERVICE_UUIDS_32_BIT_COMPLETE = 5;
    private static final int DATA_TYPE_SERVICE_UUIDS_32_BIT_PARTIAL = 4;
    private static final int DATA_TYPE_TX_POWER_LEVEL = 10;
    private static final String TAG = "ScanRecordUtil";
    private final int mAdvertiseFlags;
    private final byte[] mBytes;
    private final String mDeviceName;
    private final Map<Integer, byte[]> mManufacturerSpecificData;
    private final Map<ParcelUuid, byte[]> mServiceData;
    private final List<ParcelUuid> mServiceUuids;
    private final int mTxPowerLevel;
    static final char[] hexArray = "0123456789ABCDEF".toCharArray();
    public static final ParcelUuid BASE_UUID = ParcelUuid.fromString("00000000-0000-1000-8000-00805F9B34FB");

    private ScanRecordUtil(List<ParcelUuid> list, Map<Integer, byte[]> map, Map<ParcelUuid, byte[]> map2, int i, int i2, String str, byte[] bArr) {
        this.mServiceUuids = list;
        this.mManufacturerSpecificData = map;
        this.mServiceData = map2;
        this.mDeviceName = str;
        this.mAdvertiseFlags = i;
        this.mTxPowerLevel = i2;
        this.mBytes = bArr;
    }

    public static String bytesToHex(byte[] bArr) {
        char[] cArr = new char[bArr.length * 2];
        for (int i = 0; i < bArr.length; i++) {
            byte b = bArr[i];
            int i2 = i * 2;
            char[] cArr2 = hexArray;
            cArr[i2] = cArr2[(b & 255) >>> 4];
            cArr[i2 + 1] = cArr2[b & Ascii.SI];
        }
        return new String(cArr);
    }

    private static byte[] extractBytes(byte[] bArr, int i, int i2) {
        byte[] bArr2 = new byte[i2];
        System.arraycopy(bArr, i, bArr2, 0, i2);
        return bArr2;
    }

    /* JADX WARN: Removed duplicated region for block: B:38:0x0095  */
    /* JADX WARN: Removed duplicated region for block: B:49:0x0097  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static ScanRecordUtil parseFromBytes(byte[] bArr) {
        byte[] bArr2;
        if (bArr == null) {
            return null;
        }
        ArrayList arrayList = new ArrayList();
        ArrayMap arrayMap = new ArrayMap();
        ArrayMap arrayMap2 = new ArrayMap();
        int i = 0;
        String str = null;
        byte b = -2147483648;
        int i2 = -1;
        try {
            while (i < bArr.length) {
                try {
                    int i3 = i + 1;
                    int i4 = bArr[i] & 255;
                    if (i4 == 0) {
                        bArr2 = bArr;
                        return new ScanRecordUtil(!arrayList.isEmpty() ? null : arrayList, arrayMap, arrayMap2, i2, b, str, bArr2);
                    }
                    int i5 = i4 - 1;
                    int i6 = i + 2;
                    int i7 = bArr[i3] & 255;
                    if (i7 == 22) {
                        arrayMap2.put(parseUuidFrom(extractBytes(bArr, i6, 16)), extractBytes(bArr, i + 18, i4 - 17));
                    } else if (i7 != 255) {
                        switch (i7) {
                            case 1:
                                i2 = bArr[i6] & 255;
                                break;
                            case 2:
                            case 3:
                                parseServiceUuid(bArr, i6, i5, 16, arrayList);
                                break;
                            case 4:
                            case 5:
                                parseServiceUuid(bArr, i6, i5, 32, arrayList);
                                break;
                            case 6:
                            case 7:
                                parseServiceUuid(bArr, i6, i5, 128, arrayList);
                                break;
                            case 8:
                            case 9:
                                str = new String(extractBytes(bArr, i6, i5));
                                break;
                            case 10:
                                b = bArr[i6];
                                break;
                        }
                    } else {
                        arrayMap.put(Integer.valueOf(((bArr[i + 3] & 255) << 8) + (255 & bArr[i6])), extractBytes(bArr, i + 4, i4 - 3));
                    }
                    i = i6 + i5;
                } catch (Exception unused) {
                    bArr2 = bArr;
                }
            }
            return new ScanRecordUtil(!arrayList.isEmpty() ? null : arrayList, arrayMap, arrayMap2, i2, b, str, bArr2);
        } catch (Exception unused2) {
            return new ScanRecordUtil(null, null, null, -1, Integer.MIN_VALUE, null, bArr2);
        }
        bArr2 = bArr;
    }

    private static int parseServiceUuid(byte[] bArr, int i, int i2, int i3, List<ParcelUuid> list) {
        while (i2 > 0) {
            list.add(parseUuidFrom(extractBytes(bArr, i, i3)));
            i2 -= i3;
            i += i3;
        }
        return i;
    }

    public static ParcelUuid parseUuidFrom(byte[] bArr) {
        if (bArr == null) {
            throw new IllegalArgumentException("uuidBytes cannot be null");
        }
        int length = bArr.length;
        if (length != 16 && length != 32 && length != 128) {
            throw new IllegalArgumentException("uuidBytes length invalid - " + length);
        }
        if (length == 128) {
            ByteBuffer order = ByteBuffer.wrap(bArr).order(ByteOrder.LITTLE_ENDIAN);
            return new ParcelUuid(new UUID(order.getLong(8), order.getLong(0)));
        }
        long j = length == 16 ? (bArr[0] & 255) + ((bArr[1] & 255) << 8) : (bArr[0] & 255) + ((bArr[1] & 255) << 8) + ((bArr[2] & 255) << 16) + ((bArr[3] & 255) << 24);
        ParcelUuid parcelUuid = BASE_UUID;
        return new ParcelUuid(new UUID(parcelUuid.getUuid().getMostSignificantBits() + (j << 32), parcelUuid.getUuid().getLeastSignificantBits()));
    }

    static String toString(SparseArray<byte[]> sparseArray) {
        if (sparseArray == null) {
            return "null";
        }
        if (sparseArray.size() == 0) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        for (int i = 0; i < sparseArray.size(); i++) {
            sb.append(sparseArray.keyAt(i)).append("=").append(Arrays.toString(sparseArray.valueAt(i)));
        }
        sb.append('}');
        return sb.toString();
    }

    static <T> String toString(Map<T, byte[]> map) {
        if (map == null) {
            return "null";
        }
        if (map.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        Iterator<Map.Entry<T, byte[]>> it = map.entrySet().iterator();
        while (it.hasNext()) {
            T key = it.next().getKey();
            sb.append(key).append("=").append(Arrays.toString(map.get(key)));
            if (it.hasNext()) {
                sb.append(", ");
            }
        }
        sb.append('}');
        return sb.toString();
    }

    public int getAdvertiseFlags() {
        return this.mAdvertiseFlags;
    }

    public byte[] getBytes() {
        return this.mBytes;
    }

    public String getDeviceName() {
        return this.mDeviceName;
    }

    public Map<Integer, byte[]> getManufacturerSpecificData() {
        return this.mManufacturerSpecificData;
    }

    public byte[] getManufacturerSpecificData(int i) {
        return this.mManufacturerSpecificData.get(Integer.valueOf(i));
    }

    public Map<ParcelUuid, byte[]> getServiceData() {
        return this.mServiceData;
    }

    public byte[] getServiceData(ParcelUuid parcelUuid) {
        if (parcelUuid == null) {
            return null;
        }
        return this.mServiceData.get(parcelUuid);
    }

    public List<ParcelUuid> getServiceUuids() {
        return this.mServiceUuids;
    }

    public int getTxPowerLevel() {
        return this.mTxPowerLevel;
    }

    public String toString() {
        return "ScanRecord [mAdvertiseFlags=" + this.mAdvertiseFlags + ", mServiceUuids=" + this.mServiceUuids + ", mManufacturerSpecificData=" + toString(this.mManufacturerSpecificData) + ", mServiceData=" + toString(this.mServiceData) + ", mTxPowerLevel=" + this.mTxPowerLevel + ", mDeviceName=" + this.mDeviceName + "]";
    }
}
