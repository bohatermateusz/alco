package com.hismith.hismithapp.image_preview;

import android.app.Activity;
import android.content.Context;
import android.text.TextUtils;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.image_preview.bean.ImageInfo;
import com.hismith.hismithapp.image_preview.view.ImagePreviewActivity;
import com.hismith.hismithapp.image_preview.view.listener.OnBigImageClickListener;
import com.hismith.hismithapp.image_preview.view.listener.OnBigImageLongClickListener;
import com.hismith.hismithapp.image_preview.view.listener.OnBigImagePageChangeListener;
import com.hismith.hismithapp.image_preview.view.listener.OnOriginProgressListener;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class ImagePreview {
    private static final int MIN_DOUBLE_CLICK_TIME = 1500;
    public static final int PROGRESS_THEME_CIRCLE_TEXT = R.layout.sh_default_progress_layout;
    private OnBigImageClickListener bigImageClickListener;
    private OnBigImageLongClickListener bigImageLongClickListener;
    private OnBigImagePageChangeListener bigImagePageChangeListener;
    private WeakReference<Context> contextWeakReference;
    private List<ImageInfo> imageInfoList;
    private OnOriginProgressListener onOriginProgressListener;
    private int index = 0;
    private String folderName = "";
    private float minScale = 1.0f;
    private float mediumScale = 3.0f;
    private float maxScale = 5.0f;
    private boolean isShowIndicator = true;
    private boolean isShowCloseButton = false;
    private boolean isShowDownButton = true;
    private int zoomTransitionDuration = 200;
    private boolean isEnableDragClose = false;
    private boolean isEnableUpDragClose = false;
    private boolean isEnableClickClose = true;
    private boolean isShowErrorToast = false;
    private LoadStrategy loadStrategy = LoadStrategy.Default;
    private int indicatorShapeResId = R.drawable.shape_indicator_bg;
    private int closeIconResId = R.mipmap.ic_action_close;
    private int downIconResId = R.mipmap.icon_download_new;
    private int errorPlaceHolder = R.drawable.load_failed;
    private int progressLayoutId = -1;
    private long lastClickTime = 0;
    private int userId = 0;
    private int topicId = 0;

    private static class InnerClass {
        private static ImagePreview instance = new ImagePreview();

        private InnerClass() {
        }
    }

    public enum LoadStrategy {
        AlwaysOrigin,
        AlwaysThumb,
        NetworkAuto,
        Default
    }

    public static ImagePreview getInstance() {
        return InnerClass.instance;
    }

    private ImagePreview setOnOriginProgressListener(OnOriginProgressListener onOriginProgressListener) {
        this.onOriginProgressListener = onOriginProgressListener;
        return this;
    }

    public OnBigImageClickListener getBigImageClickListener() {
        return this.bigImageClickListener;
    }

    public OnBigImageLongClickListener getBigImageLongClickListener() {
        return this.bigImageLongClickListener;
    }

    public OnBigImagePageChangeListener getBigImagePageChangeListener() {
        return this.bigImagePageChangeListener;
    }

    public int getCloseIconResId() {
        return this.closeIconResId;
    }

    public int getDownIconResId() {
        return this.downIconResId;
    }

    public int getErrorPlaceHolder() {
        return this.errorPlaceHolder;
    }

    public String getFolderName() {
        if (TextUtils.isEmpty(this.folderName)) {
            this.folderName = "Download";
        }
        return this.folderName;
    }

    public List<ImageInfo> getImageInfoList() {
        return this.imageInfoList;
    }

    public int getIndex() {
        return this.index;
    }

    public int getIndicatorShapeResId() {
        return this.indicatorShapeResId;
    }

    public LoadStrategy getLoadStrategy() {
        return this.loadStrategy;
    }

    public float getMaxScale() {
        return this.maxScale;
    }

    public float getMediumScale() {
        return this.mediumScale;
    }

    public float getMinScale() {
        return this.minScale;
    }

    public OnOriginProgressListener getOnOriginProgressListener() {
        return this.onOriginProgressListener;
    }

    public int getProgressLayoutId() {
        return this.progressLayoutId;
    }

    public int getTopicId() {
        return this.topicId;
    }

    public int getUserId() {
        return this.userId;
    }

    public int getZoomTransitionDuration() {
        return this.zoomTransitionDuration;
    }

    public boolean isEnableClickClose() {
        return this.isEnableClickClose;
    }

    public boolean isEnableDragClose() {
        return this.isEnableDragClose;
    }

    public boolean isEnableUpDragClose() {
        return this.isEnableUpDragClose;
    }

    public boolean isShowCloseButton() {
        return this.isShowCloseButton;
    }

    public boolean isShowDownButton() {
        return this.isShowDownButton;
    }

    public boolean isShowErrorToast() {
        return this.isShowErrorToast;
    }

    public boolean isShowIndicator() {
        return this.isShowIndicator;
    }

    public boolean isShowOriginButton(int i) {
        List<ImageInfo> imageInfoList = getImageInfoList();
        if (imageInfoList == null || imageInfoList.size() == 0 || imageInfoList.get(i).getOriginUrl().equalsIgnoreCase(imageInfoList.get(i).getThumbnailUrl())) {
            return false;
        }
        if (this.loadStrategy == LoadStrategy.Default) {
            return true;
        }
        if (this.loadStrategy == LoadStrategy.NetworkAuto || this.loadStrategy == LoadStrategy.AlwaysThumb) {
            return false;
        }
        LoadStrategy loadStrategy = LoadStrategy.AlwaysOrigin;
        return false;
    }

    public void reset() {
        this.imageInfoList = null;
        this.index = 0;
        this.userId = 0;
        this.topicId = 0;
        this.minScale = 1.0f;
        this.mediumScale = 3.0f;
        this.maxScale = 5.0f;
        this.zoomTransitionDuration = 200;
        this.isShowDownButton = true;
        this.isShowCloseButton = false;
        this.isEnableDragClose = false;
        this.isEnableClickClose = true;
        this.isShowIndicator = true;
        this.isShowErrorToast = false;
        this.closeIconResId = R.mipmap.ic_action_close;
        this.downIconResId = R.mipmap.icon_download_new;
        this.errorPlaceHolder = R.drawable.load_failed;
        this.loadStrategy = LoadStrategy.Default;
        this.folderName = "Download";
        WeakReference<Context> weakReference = this.contextWeakReference;
        if (weakReference != null) {
            weakReference.clear();
            this.contextWeakReference = null;
        }
        this.bigImageClickListener = null;
        this.bigImageLongClickListener = null;
        this.bigImagePageChangeListener = null;
        this.progressLayoutId = -1;
        this.lastClickTime = 0L;
    }

    public ImagePreview setBigImageClickListener(OnBigImageClickListener onBigImageClickListener) {
        this.bigImageClickListener = onBigImageClickListener;
        return this;
    }

    public ImagePreview setBigImageLongClickListener(OnBigImageLongClickListener onBigImageLongClickListener) {
        this.bigImageLongClickListener = onBigImageLongClickListener;
        return this;
    }

    public ImagePreview setBigImagePageChangeListener(OnBigImagePageChangeListener onBigImagePageChangeListener) {
        this.bigImagePageChangeListener = onBigImagePageChangeListener;
        return this;
    }

    public ImagePreview setCloseIconResId(int i) {
        this.closeIconResId = i;
        return this;
    }

    public ImagePreview setContext(Context context) {
        this.contextWeakReference = new WeakReference<>(context);
        return this;
    }

    public ImagePreview setDownIconResId(int i) {
        this.downIconResId = i;
        return this;
    }

    public ImagePreview setEnableClickClose(boolean z) {
        this.isEnableClickClose = z;
        return this;
    }

    public ImagePreview setEnableDragClose(boolean z) {
        this.isEnableDragClose = z;
        return this;
    }

    public ImagePreview setEnableUpDragClose(boolean z) {
        this.isEnableUpDragClose = z;
        return this;
    }

    public ImagePreview setErrorPlaceHolder(int i) {
        this.errorPlaceHolder = i;
        return this;
    }

    public ImagePreview setFolderName(String str) {
        this.folderName = str;
        return this;
    }

    public ImagePreview setImage(String str) {
        this.imageInfoList = new ArrayList();
        ImageInfo imageInfo = new ImageInfo();
        imageInfo.setThumbnailUrl(str);
        imageInfo.setOriginUrl(str);
        this.imageInfoList.add(imageInfo);
        return this;
    }

    public ImagePreview setImageInfoList(List<ImageInfo> list) {
        this.imageInfoList = list;
        return this;
    }

    public ImagePreview setImageList(List<String> list) {
        this.imageInfoList = new ArrayList();
        for (int i = 0; i < list.size(); i++) {
            ImageInfo imageInfo = new ImageInfo();
            imageInfo.setThumbnailUrl(list.get(i));
            imageInfo.setOriginUrl(list.get(i));
            this.imageInfoList.add(imageInfo);
        }
        return this;
    }

    public ImagePreview setIndex(int i) {
        this.index = i;
        return this;
    }

    public ImagePreview setIndicatorShapeResId(int i) {
        this.indicatorShapeResId = i;
        return this;
    }

    public ImagePreview setLoadStrategy(LoadStrategy loadStrategy) {
        this.loadStrategy = loadStrategy;
        return this;
    }

    public ImagePreview setProgressLayoutId(int i, OnOriginProgressListener onOriginProgressListener) {
        setOnOriginProgressListener(onOriginProgressListener);
        this.progressLayoutId = i;
        return this;
    }

    @Deprecated
    public ImagePreview setScaleLevel(int i, int i2, int i3) {
        if (i3 <= i2 || i2 <= i || i <= 0) {
            throw new IllegalArgumentException("max must greater to medium, medium must greater to min!");
        }
        this.minScale = i;
        this.mediumScale = i2;
        this.maxScale = i3;
        return this;
    }

    @Deprecated
    public ImagePreview setScaleMode(int i) {
        return this;
    }

    public ImagePreview setShowCloseButton(boolean z) {
        this.isShowCloseButton = z;
        return this;
    }

    public ImagePreview setShowDownButton(boolean z) {
        this.isShowDownButton = z;
        return this;
    }

    public ImagePreview setShowErrorToast(boolean z) {
        this.isShowErrorToast = z;
        return this;
    }

    public ImagePreview setShowIndicator(boolean z) {
        this.isShowIndicator = z;
        return this;
    }

    @Deprecated
    public ImagePreview setShowOriginButton(boolean z) {
        return this;
    }

    public ImagePreview setTopicId(int i) {
        this.topicId = i;
        return this;
    }

    public ImagePreview setUserId(int i) {
        this.userId = i;
        return this;
    }

    public ImagePreview setZoomTransitionDuration(int i) {
        if (i < 0) {
            throw new IllegalArgumentException("zoomTransitionDuration must greater 0");
        }
        this.zoomTransitionDuration = i;
        return this;
    }

    public void start() {
        if (System.currentTimeMillis() - this.lastClickTime <= 1500) {
            return;
        }
        WeakReference<Context> weakReference = this.contextWeakReference;
        if (weakReference == null) {
            throw new IllegalArgumentException("You must call 'setContext(Context context)' first!");
        }
        Context context = weakReference.get();
        if (context == null) {
            throw new IllegalArgumentException("You must call 'setContext(Context context)' first!");
        }
        if (!(context instanceof Activity)) {
            throw new IllegalArgumentException("context must be a Activity!");
        }
        Activity activity = (Activity) context;
        if (activity.isFinishing() || activity.isDestroyed()) {
            reset();
            return;
        }
        List<ImageInfo> list = this.imageInfoList;
        if (list == null || list.size() == 0) {
            throw new IllegalArgumentException("Do you forget to call 'setImageInfoList(List<ImageInfo> imageInfoList)' ?");
        }
        if (this.index >= this.imageInfoList.size()) {
            throw new IllegalArgumentException("index out of range!");
        }
        this.lastClickTime = System.currentTimeMillis();
        ImagePreviewActivity.activityStart(context);
    }
}
