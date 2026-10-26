package com.hismith.hismithapp.filter;

import android.text.InputFilter;
import android.text.Spanned;
import com.bumptech.glide.load.Key;
import java.io.UnsupportedEncodingException;

/* loaded from: classes3.dex */
public class BytesLengthFilter implements InputFilter {
    private final int maxBytes;

    public BytesLengthFilter(int i) {
        this.maxBytes = i;
    }

    private String truncateToBytes(String str, int i) throws UnsupportedEncodingException {
        StringBuilder sb = new StringBuilder();
        int i2 = 0;
        int i3 = 0;
        while (i2 < str.length()) {
            int codePointAt = Character.codePointAt(str, i2);
            int charCount = Character.charCount(codePointAt);
            String str2 = new String(Character.toChars(codePointAt));
            i3 += str2.getBytes(Key.STRING_CHARSET_NAME).length;
            if (i3 > i) {
                break;
            }
            sb.append(str2);
            i2 += charCount;
        }
        return sb.toString();
    }

    @Override // android.text.InputFilter
    public CharSequence filter(CharSequence charSequence, int i, int i2, Spanned spanned, int i3, int i4) {
        if (charSequence.length() == 0) {
            return null;
        }
        try {
            int length = spanned.subSequence(0, i3).toString().getBytes(Key.STRING_CHARSET_NAME).length;
            int length2 = (this.maxBytes - length) - spanned.subSequence(i4, spanned.length()).toString().getBytes(Key.STRING_CHARSET_NAME).length;
            if (length2 <= 0) {
                return "";
            }
            String obj = charSequence.subSequence(i, i2).toString();
            if (obj.getBytes(Key.STRING_CHARSET_NAME).length <= length2) {
                return null;
            }
            return truncateToBytes(obj, length2);
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
            return "";
        }
    }
}
