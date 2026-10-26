package com.hismith.hismithapp.manager;

import android.net.Uri;
import android.os.Build;
import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import com.hismith.hismithapp.picker.PictureSelectorManager;
import com.luck.picture.lib.entity.LocalMedia;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class PictureResultManager {
    private final ComponentActivity activity;
    private ActivityResultLauncher<PickVisualMediaRequest> launcher;
    private final Map<RequestKey, OnGalleryListener> listenerMap = new HashMap();
    private RequestKey currentRequestKey = RequestKey.DEFAULT;

    public interface OnGalleryListener {
        void onGalleryResult(Uri uri);

        void onGalleryResultFile(String str);
    }

    public enum RequestKey {
        AVATAR,
        COVER,
        BANNER,
        DEFAULT
    }

    public PictureResultManager(ComponentActivity componentActivity) {
        this.activity = componentActivity;
        initLauncher();
    }

    private void initLauncher() {
        this.launcher = this.activity.registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), new ActivityResultCallback() { // from class: com.hismith.hismithapp.manager.PictureResultManager$$ExternalSyntheticLambda0
            @Override // androidx.activity.result.ActivityResultCallback
            public final void onActivityResult(Object obj) {
                PictureResultManager.this.m610xaa4bdbe((Uri) obj);
            }
        });
    }

    public static boolean isGoogleCoulePhotosUri(Uri uri) {
        if (uri != null && "com.google.android.apps.photos.contentprovider".equals(uri.getAuthority())) {
            List<String> pathSegments = uri.getPathSegments();
            if (pathSegments.size() > 2 && pathSegments.get(2).contains("mediakey:/")) {
                return true;
            }
        }
        return false;
    }

    /* renamed from: lambda$initLauncher$0$com-hismith-hismithapp-manager-PictureResultManager, reason: not valid java name */
    /* synthetic */ void m610xaa4bdbe(Uri uri) {
        OnGalleryListener onGalleryListener = this.listenerMap.get(this.currentRequestKey);
        if (onGalleryListener != null) {
            onGalleryListener.onGalleryResult(uri);
            this.listenerMap.remove(this.currentRequestKey);
        }
    }

    public void openGallery(OnGalleryListener onGalleryListener) {
        openGallery(RequestKey.DEFAULT, onGalleryListener);
    }

    public void openGallery(RequestKey requestKey, final OnGalleryListener onGalleryListener) {
        if (this.launcher == null) {
            throw new IllegalStateException("Launcher 未初始化，请在 Activity.onCreate() 创建 PictureResultManager 实例");
        }
        if (Build.VERSION.SDK_INT < 33) {
            PictureSelectorManager.getInstance(this.activity).openGallert(new PictureSelectorManager.OnGallertListener() { // from class: com.hismith.hismithapp.manager.PictureResultManager.1
                @Override // com.hismith.hismithapp.picker.PictureSelectorManager.OnGallertListener
                public void onGallertResult(List<LocalMedia> list) {
                    for (int i = 0; i < list.size(); i++) {
                        String compressPath = list.get(i).isCompressed() ? list.get(i).getCompressPath() : list.get(i).getRealPath();
                        OnGalleryListener onGalleryListener2 = onGalleryListener;
                        if (onGalleryListener2 != null) {
                            onGalleryListener2.onGalleryResultFile(compressPath);
                        }
                    }
                }
            });
            return;
        }
        this.currentRequestKey = requestKey;
        this.listenerMap.put(requestKey, onGalleryListener);
        this.launcher.launch(new PickVisualMediaRequest.Builder().setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE).build());
    }
}
