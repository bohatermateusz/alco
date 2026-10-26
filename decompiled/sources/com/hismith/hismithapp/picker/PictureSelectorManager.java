package com.hismith.hismithapp.picker;

import android.app.Activity;
import android.content.Context;
import androidx.fragment.app.Fragment;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.luck.picture.lib.basic.PictureSelector;
import com.luck.picture.lib.config.SelectMimeType;
import com.luck.picture.lib.entity.LocalMedia;
import com.luck.picture.lib.interfaces.OnResultCallbackListener;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PictureSelectorManager {
    private static PictureSelectorManager instance;
    private static int mLanguage;
    private Activity mActivity;

    public interface OnCameraListener {
        void onCameraResult(List<LocalMedia> list);
    }

    public interface OnGallertListener {
        void onGallertResult(List<LocalMedia> list);
    }

    public PictureSelectorManager(Activity activity) {
        this.mActivity = activity;
        refreshLanguage(activity);
    }

    public static PictureSelectorManager getInstance(Activity activity) {
        PictureSelectorManager pictureSelectorManager;
        synchronized (PictureSelectorManager.class) {
            pictureSelectorManager = new PictureSelectorManager(activity);
            instance = pictureSelectorManager;
        }
        return pictureSelectorManager;
    }

    public static PictureSelectorManager getInstance(Fragment fragment) {
        return getInstance(fragment.getActivity());
    }

    public static void refreshLanguage(Context context) {
        if (context == null) {
        }
        String currLanguage = LanguageUtil.getCurrLanguage(context);
        currLanguage.hashCode();
        switch (currLanguage) {
            case "de":
                mLanguage = 4;
                break;
            case "en":
            case "nl":
                mLanguage = 2;
                break;
            case "fr":
                mLanguage = 5;
                break;
            case "ja":
                mLanguage = 6;
                break;
            default:
                mLanguage = 2;
                break;
        }
    }

    public void openCamera(final OnCameraListener onCameraListener) {
        PictureSelector.create(this.mActivity).openCamera(SelectMimeType.ofImage()).forResult(new OnResultCallbackListener<LocalMedia>() { // from class: com.hismith.hismithapp.picker.PictureSelectorManager.2
            @Override // com.luck.picture.lib.interfaces.OnResultCallbackListener
            public void onCancel() {
            }

            @Override // com.luck.picture.lib.interfaces.OnResultCallbackListener
            public void onResult(ArrayList<LocalMedia> arrayList) {
                OnCameraListener onCameraListener2 = onCameraListener;
                if (onCameraListener2 != null) {
                    onCameraListener2.onCameraResult(arrayList);
                }
            }
        });
    }

    public void openGallert(final OnGallertListener onGallertListener) {
        PictureSelector.create(this.mActivity).openSystemGallery(SelectMimeType.ofImage()).setSelectionMode(1).forSystemResult(new OnResultCallbackListener<LocalMedia>() { // from class: com.hismith.hismithapp.picker.PictureSelectorManager.1
            @Override // com.luck.picture.lib.interfaces.OnResultCallbackListener
            public void onCancel() {
            }

            @Override // com.luck.picture.lib.interfaces.OnResultCallbackListener
            public void onResult(ArrayList<LocalMedia> arrayList) {
                OnGallertListener onGallertListener2 = onGallertListener;
                if (onGallertListener2 != null) {
                    onGallertListener2.onGallertResult(arrayList);
                }
            }
        });
    }

    public void openGallert2(OnGallertListener onGallertListener) {
    }
}
