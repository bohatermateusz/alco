package com.hismith.hismithapp.bluetooth;

import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanSettings;
import android.os.ParcelUuid;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class BleScanFilter {
    private ScanFilter.Builder scanFilterBuilder;
    private List<ScanFilter> scanFilterList;
    private ScanSettings.Builder scanSettingBuilder;

    private List<ScanFilter> buildScanFilters() {
        this.scanFilterList = new ArrayList();
        this.scanFilterBuilder = new ScanFilter.Builder();
        ParcelUuid fromString = ParcelUuid.fromString("FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF");
        this.scanFilterBuilder.setServiceUuid(ParcelUuid.fromString("0000ffe5-0000-1000-8000-00805f9b34fb"), fromString);
        this.scanFilterList.add(this.scanFilterBuilder.build());
        return this.scanFilterList;
    }

    private ScanSettings buildScanSettings() {
        ScanSettings.Builder builder = new ScanSettings.Builder();
        this.scanSettingBuilder = builder;
        builder.setScanMode(2);
        this.scanSettingBuilder.setMatchMode(1);
        this.scanSettingBuilder.setCallbackType(1);
        return this.scanSettingBuilder.build();
    }
}
