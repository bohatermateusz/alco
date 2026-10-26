package com.hismith.hismithapp.image_preview.bean;

import java.io.Serializable;

/* loaded from: classes3.dex */
public class ImageInfo implements Serializable {
    private String originUrl;
    private String thumbnailUrl;

    public String getOriginUrl() {
        return this.originUrl;
    }

    public String getThumbnailUrl() {
        return this.thumbnailUrl;
    }

    public void setOriginUrl(String str) {
        this.originUrl = str;
    }

    public void setThumbnailUrl(String str) {
        this.thumbnailUrl = str;
    }
}
