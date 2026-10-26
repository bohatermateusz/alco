package com.hismith.hismithapp.bluetooth;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/* loaded from: classes3.dex */
public class ParsedAd {
    public String diyInfo;
    public byte flags;
    public String localName;
    public short manufacturer;
    public List<UUID> uuids = new ArrayList();
}
