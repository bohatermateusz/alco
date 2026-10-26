package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.text.Editable;
import android.text.SpannableString;
import android.text.TextUtils;
import android.text.style.ImageSpan;
import android.util.AttributeSet;
import androidx.appcompat.widget.AppCompatEditText;
import com.hismith.hismithapp.utils.ImageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.RequestBodyUtil;
import com.luck.picture.lib.config.PictureMimeType;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/* loaded from: classes3.dex */
public class PictureAndTextEditorView extends AppCompatEditText {
    public static final String mBitmapEndTag = "</img>";
    public static final String mBitmapTag = "<img>";
    private final String TAG;
    private Context mContext;
    private List<String> mImageList;
    private String mNewLineTag;

    public interface onTextImageCallback {
        void onImagePathCallback(String str);
    }

    public PictureAndTextEditorView(Context context) {
        super(context);
        this.TAG = "PATEditorView";
        this.mNewLineTag = "\n";
        init(context);
    }

    public PictureAndTextEditorView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.TAG = "PATEditorView";
        this.mNewLineTag = "\n";
        init(context);
    }

    public PictureAndTextEditorView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.TAG = "PATEditorView";
        this.mNewLineTag = "\n";
        init(context);
    }

    public static String buildContentNameToPath(String str, String[] strArr) {
        StringBuilder sb = new StringBuilder();
        if (!TextUtils.isEmpty(str) && strArr != null) {
            String[] split = str.split("(?=<img>)|(?<=</img>)");
            for (int i = 0; i < split.length; i++) {
                if (split[i].startsWith(mBitmapTag)) {
                    String substring = split[i].substring(mBitmapTag.length(), split[i].indexOf(mBitmapEndTag));
                    for (int i2 = 0; i2 < strArr.length; i2++) {
                        File file = new File(strArr[i2]);
                        if (file.exists() && file.getName().equals(substring)) {
                            sb.append(mBitmapTag);
                            sb.append(strArr[i2]);
                            sb.append(mBitmapEndTag);
                        }
                    }
                } else {
                    sb.append(split[i]);
                }
            }
        }
        return TextUtils.isEmpty(sb.toString()) ? str : sb.toString();
    }

    private void init(Context context) {
        this.mContext = context;
        this.mImageList = new ArrayList();
    }

    private SpannableString insertBitmap(String str, Bitmap bitmap, boolean z) {
        Editable editableText = getEditableText();
        int selectionStart = getSelectionStart();
        SpannableString spannableString = new SpannableString(z ? this.mNewLineTag : "");
        String str2 = mBitmapTag + str + mBitmapEndTag;
        SpannableString spannableString2 = new SpannableString(str2);
        spannableString2.setSpan(new ImageSpan(this.mContext, bitmap), 0, str2.length(), 33);
        if (selectionStart < 0 || selectionStart >= editableText.length()) {
            editableText.append((CharSequence) spannableString2);
        } else {
            editableText.insert(selectionStart, spannableString2);
        }
        editableText.insert(selectionStart, spannableString);
        LogUtil.e("插入图片name:" + str2 + ", getEditableText():" + ((Object) getEditableText()));
        return spannableString2;
    }

    public Bitmap calculateBitmap(Bitmap bitmap, float f) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        if (width == 0 || height == 0) {
            return bitmap;
        }
        return ThumbnailUtils.extractThumbnail(bitmap, (int) f, (int) ((height * f) / width));
    }

    public int calculateInSampleSize(BitmapFactory.Options options, int i, int i2) {
        int i3 = options.outHeight;
        int i4 = options.outWidth;
        if (i3 <= i2 && i4 <= i) {
            return 1;
        }
        int round = Math.round(i3 / i2);
        int round2 = Math.round(i4 / i);
        return round < round2 ? round : round2;
    }

    public String getContentText() {
        String replaceAll = getText().toString().replaceAll(this.mNewLineTag, "");
        Matcher matcher = Pattern.compile("<img>(.*?)</img>").matcher(replaceAll);
        while (matcher.find()) {
            replaceAll = replaceAll.replace(matcher.group(), "");
        }
        return replaceAll;
    }

    public String[] getImageList() {
        return (String[]) this.mImageList.toArray(new String[0]);
    }

    public Bitmap getSmallBitmap(String str, int i, int i2) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(str, options);
        options.inSampleSize = calculateInSampleSize(options, i, i2);
        options.inJustDecodeBounds = false;
        Bitmap decodeFile = BitmapFactory.decodeFile(str, options);
        if (decodeFile == null) {
            return null;
        }
        int i3 = (int) (this.mContext.getResources().getDisplayMetrics().widthPixels * 0.97d);
        return Bitmap.createScaledBitmap(decodeFile, i3, (decodeFile.getHeight() * i3) / decodeFile.getWidth(), false);
    }

    public void insertBitmap(String str, boolean z) {
        File file = new File(str);
        if (file.exists()) {
            int width = getWidth();
            LogUtil.e("width:" + width);
            this.mImageList.add(str);
            Bitmap image = ImageUtil.getImage(str);
            if (image != null) {
                image = calculateBitmap(image, width);
            }
            if (image != null) {
                insertBitmap(file.getName(), image, z);
            }
        }
    }

    public void insertUriBitmap(Uri uri, boolean z) {
        if (uri != null) {
            try {
                int width = getWidth();
                Bitmap decodeStream = BitmapFactory.decodeStream(this.mContext.getContentResolver().openInputStream(uri));
                if (decodeStream != null) {
                    decodeStream = calculateBitmap(decodeStream, width);
                }
                String fileNameFromUri = RequestBodyUtil.getFileNameFromUri(this.mContext, uri);
                if (decodeStream != null) {
                    insertBitmap(fileNameFromUri, decodeStream, z);
                }
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void setText(String str, onTextImageCallback ontextimagecallback) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        String[] split = str.split("(?=<img>)|(?<=</img>)");
        LogUtil.e(Arrays.toString(split));
        for (int i = 0; i < split.length; i++) {
            if (split[i].startsWith(mBitmapTag)) {
                String substring = split[i].substring(mBitmapTag.length(), split[i].indexOf(mBitmapEndTag));
                if (PictureMimeType.isContent(substring)) {
                    Uri parse = Uri.parse(substring);
                    if (parse != null) {
                        insertUriBitmap(parse, false);
                    }
                } else if (new File(substring).exists()) {
                    insertBitmap(substring, false);
                }
                if (ontextimagecallback != null) {
                    ontextimagecallback.onImagePathCallback(substring);
                }
            } else {
                LogUtil.e("append:" + split[i]);
                append(split[i]);
            }
        }
    }
}
