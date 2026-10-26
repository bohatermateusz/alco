package com.hismith.hismithapp.image_preview.tool.image;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.image_preview.glide.FileTarget;
import com.hismith.hismithapp.image_preview.tool.file.FileUtil;
import com.hismith.hismithapp.image_preview.tool.file.SingleMediaScanner;
import com.hismith.hismithapp.image_preview.tool.text.MD5Util;
import com.hismith.hismithapp.image_preview.tool.ui.ToastUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;

/* loaded from: classes3.dex */
public class DownloadPictureUtil {
    public static void downloadPicture(final Context context, final String str) {
        Glide.with(context).downloadOnly().load(str).into((RequestBuilder<File>) new FileTarget() { // from class: com.hismith.hismithapp.image_preview.tool.image.DownloadPictureUtil.1
            @Override // com.hismith.hismithapp.image_preview.glide.FileTarget, com.bumptech.glide.request.target.Target
            public void onLoadFailed(Drawable drawable) {
                super.onLoadFailed(drawable);
                ToastUtil.getInstance()._short(context, LanguageUtil.getLanguageByKey(context, "saveFailed"));
            }

            @Override // com.hismith.hismithapp.image_preview.glide.FileTarget, com.bumptech.glide.request.target.Target
            public void onLoadStarted(Drawable drawable) {
                super.onLoadStarted(drawable);
                ToastUtil.getInstance()._short(context, LanguageUtil.getLanguageByKey(context, "startDownload"));
                super.onLoadStarted(drawable);
            }

            /* JADX WARN: Can't rename method to resolve collision */
            @Override // com.hismith.hismithapp.image_preview.glide.FileTarget, com.bumptech.glide.request.target.Target
            public void onResourceReady(File file, Transition<? super File> transition) {
                String str2;
                OutputStream openOutputStream;
                super.onResourceReady(file, transition);
                String folderName = ImagePreview.getInstance().getFolderName();
                try {
                    String str3 = str;
                    String substring = str3.substring(str3.lastIndexOf("/") + 1);
                    if (substring.contains(".")) {
                        substring = substring.substring(0, substring.lastIndexOf("."));
                    }
                    str2 = MD5Util.md5Encode(substring);
                } catch (Exception e) {
                    e.printStackTrace();
                    str2 = System.currentTimeMillis() + "";
                }
                String imageTypeWithMime = ImageUtil.getImageTypeWithMime(file.getAbsolutePath());
                String str4 = str2 + "." + imageTypeWithMime;
                if (Build.VERSION.SDK_INT < 29) {
                    String str5 = Environment.getExternalStorageDirectory() + "/" + folderName + "/";
                    FileUtil.createFileByDeleteOldFile(str5 + str4);
                    boolean copyFile = FileUtil.copyFile(file, str5, str4);
                    Context context2 = context;
                    if (!copyFile) {
                        ToastUtil.getInstance()._short(context, LanguageUtil.getLanguageByKey(context2, "saveFailed"));
                        return;
                    } else {
                        ToastUtil.getInstance()._short(context, (LanguageUtil.getLanguageByKey(context2, "successfullySavedTo") + " ").concat(str5));
                        new SingleMediaScanner(context, str5.concat(str4), new SingleMediaScanner.ScanListener() { // from class: com.hismith.hismithapp.image_preview.tool.image.DownloadPictureUtil.1.1
                            @Override // com.hismith.hismithapp.image_preview.tool.file.SingleMediaScanner.ScanListener
                            public void onScanFinish() {
                            }
                        });
                        return;
                    }
                }
                ContentResolver contentResolver = context.getContentResolver();
                ContentValues contentValues = new ContentValues();
                contentValues.put("_display_name", str4);
                contentValues.put("description", str4);
                contentValues.put("mime_type", "image/" + imageTypeWithMime);
                contentValues.put("relative_path", Environment.DIRECTORY_PICTURES + "/" + folderName + "/");
                contentValues.put("is_pending", (Integer) 1);
                Uri insert = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
                try {
                    BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(file.getAbsolutePath()));
                    if (insert != null) {
                        try {
                            openOutputStream = contentResolver.openOutputStream(insert);
                        } finally {
                        }
                    } else {
                        openOutputStream = null;
                    }
                    if (openOutputStream != null) {
                        try {
                            byte[] bArr = new byte[4096];
                            while (true) {
                                int read = bufferedInputStream.read(bArr);
                                if (read == -1) {
                                    break;
                                } else {
                                    openOutputStream.write(bArr, 0, read);
                                }
                            }
                            openOutputStream.flush();
                        } finally {
                        }
                    }
                    contentValues.clear();
                    contentValues.put("is_pending", (Integer) 0);
                    contentResolver.update(insert, contentValues, null, null);
                    ToastUtil.getInstance()._short(context, (LanguageUtil.getLanguageByKey(context, "successfullySavedTo") + " ").concat(Environment.DIRECTORY_PICTURES + "/" + folderName));
                    if (openOutputStream != null) {
                        openOutputStream.close();
                    }
                    bufferedInputStream.close();
                } catch (IOException e2) {
                    e2.printStackTrace();
                    ToastUtil.getInstance()._short(context, LanguageUtil.getLanguageByKey(context, "saveFailed"));
                }
            }
        });
    }
}
