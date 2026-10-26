package com.hismith.hismithapp.device_rule;

import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.utils.LanguageUtil;

/* loaded from: classes3.dex */
public enum ConditionType {
    GREATER_THAN(LanguageUtil.getLanguageByKey(MyApplication.getApplication(), "greater")),
    EQUAL(LanguageUtil.getLanguageByKey(MyApplication.getApplication(), "equal")),
    LESS_THAN(LanguageUtil.getLanguageByKey(MyApplication.getApplication(), "less"));

    private final String displayName;

    ConditionType(String str) {
        this.displayName = str;
    }

    public String getDisplayName() {
        return this.displayName;
    }

    @Override // java.lang.Enum
    public String toString() {
        return this.displayName;
    }
}
