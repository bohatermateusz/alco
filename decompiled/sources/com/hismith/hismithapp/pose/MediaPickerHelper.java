package com.hismith.hismithapp.pose;

import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.luck.picture.lib.config.PictureMimeType;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/* loaded from: classes3.dex */
public class MediaPickerHelper {
    private final ComponentActivity activity;
    private final Callback callback;
    private ActivityResultLauncher<PickVisualMediaRequest> pickerLauncher;

    public interface Callback {
        void onError(Exception exc);

        void onImagePicked(Bitmap bitmap, Uri uri);

        void onVideoPicked(File file, Uri uri);
    }

    public MediaPickerHelper(ComponentActivity componentActivity, Callback callback) {
        this.activity = componentActivity;
        this.callback = callback;
        init();
    }

    private File copyUriToFile(Uri uri) throws IOException {
        String fileName = getFileName(uri);
        if (fileName == null) {
            fileName = "temp_video_" + System.currentTimeMillis() + PictureMimeType.MP4;
        }
        File file = new File(this.activity.getCacheDir(), fileName);
        InputStream openInputStream = this.activity.getContentResolver().openInputStream(uri);
        FileOutputStream fileOutputStream = new FileOutputStream(file);
        byte[] bArr = new byte[4096];
        while (true) {
            int read = openInputStream.read(bArr);
            if (read <= 0) {
                openInputStream.close();
                fileOutputStream.close();
                return file;
            }
            fileOutputStream.write(bArr, 0, read);
        }
    }

    private String getFileName(Uri uri) {
        Uri uri2;
        int columnIndex;
        String str = null;
        if (FirebaseAnalytics.Param.CONTENT.equals(uri.getScheme())) {
            uri2 = uri;
            Cursor query = this.activity.getContentResolver().query(uri2, null, null, null, null);
            if (query != null) {
                try {
                    if (query.moveToFirst() && (columnIndex = query.getColumnIndex("_display_name")) >= 0) {
                        str = query.getString(columnIndex);
                    }
                } finally {
                }
            }
            if (query != null) {
                query.close();
            }
        } else {
            uri2 = uri;
        }
        return str == null ? uri2.getLastPathSegment() : str;
    }

    private void handleImage(Uri uri) {
        try {
            int i = Build.VERSION.SDK_INT;
            ComponentActivity componentActivity = this.activity;
            this.callback.onImagePicked((i >= 28 ? ImageDecoder.decodeBitmap(ImageDecoder.createSource(componentActivity.getContentResolver(), uri)) : MediaStore.Images.Media.getBitmap(componentActivity.getContentResolver(), uri)).copy(Bitmap.Config.ARGB_8888, true), uri);
        } catch (Exception e) {
            this.callback.onError(e);
        }
    }

    private void handleVideo(Uri uri) {
        try {
            this.callback.onVideoPicked(copyUriToFile(uri), uri);
        } catch (Exception e) {
            this.callback.onError(e);
        }
    }

    private void init() {
        this.pickerLauncher = this.activity.registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), new ActivityResultCallback() { // from class: com.hismith.hismithapp.pose.MediaPickerHelper$$ExternalSyntheticLambda0
            @Override // androidx.activity.result.ActivityResultCallback
            public final void onActivityResult(Object obj) {
                MediaPickerHelper.this.m615lambda$init$0$comhismithhismithappposeMediaPickerHelper((Uri) obj);
            }
        });
    }

    /* renamed from: lambda$init$0$com-hismith-hismithapp-pose-MediaPickerHelper, reason: not valid java name */
    /* synthetic */ void m615lambda$init$0$comhismithhismithappposeMediaPickerHelper(Uri uri) {
        if (uri == null) {
            return;
        }
        String type = this.activity.getContentResolver().getType(uri);
        if (type != null && type.startsWith("image")) {
            handleImage(uri);
        } else {
            if (type == null || !type.startsWith("video")) {
                return;
            }
            handleVideo(uri);
        }
    }

    public void pickImage() {
        this.pickerLauncher.launch(new PickVisualMediaRequest.Builder().setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE).build());
    }

    public void pickVideo() {
        this.pickerLauncher.launch(new PickVisualMediaRequest.Builder().setMediaType(ActivityResultContracts.PickVisualMedia.VideoOnly.INSTANCE).build());
    }
}
