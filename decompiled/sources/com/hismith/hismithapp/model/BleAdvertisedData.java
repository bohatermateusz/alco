package com.hismith.hismithapp.model;

import java.util.List;
import java.util.UUID;

/* loaded from: classes3.dex */
public class BleAdvertisedData {
    private String mName;
    private List<UUID> mUuids;

    public BleAdvertisedData(List<UUID> list, String str) {
        this.mUuids = list;
        this.mName = str;
    }

    public String getName() {
        return this.mName;
    }

    public List<UUID> getUuids() {
        return this.mUuids;
    }
}
