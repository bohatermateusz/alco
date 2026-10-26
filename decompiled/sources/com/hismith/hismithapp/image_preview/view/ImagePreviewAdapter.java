package com.hismith.hismithapp.image_preview.view;

import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager.widget.PagerAdapter;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.load.DataSource;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.bumptech.glide.load.engine.GlideException;
import com.bumptech.glide.load.resource.gif.GifDrawable;
import com.bumptech.glide.request.BaseRequestOptions;
import com.bumptech.glide.request.RequestListener;
import com.bumptech.glide.request.RequestOptions;
import com.bumptech.glide.request.target.Target;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.image_preview.bean.ImageInfo;
import com.hismith.hismithapp.image_preview.glide.FileTarget;
import com.hismith.hismithapp.image_preview.glide.ImageLoader;
import com.hismith.hismithapp.image_preview.tool.common.NetworkUtil;
import com.hismith.hismithapp.image_preview.tool.image.ImageUtil;
import com.hismith.hismithapp.image_preview.tool.ui.PhoneUtil;
import com.hismith.hismithapp.image_preview.tool.ui.ToastUtil;
import com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper;
import com.hismith.hismithapp.image_preview.view.helper.ImageSource;
import com.hismith.hismithapp.image_preview.view.helper.SubsamplingScaleImageViewDragClose;
import com.hismith.hismithapp.image_preview.view.photoview.PhotoView;
import com.hismith.hismithapp.utils.LanguageUtil;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class ImagePreviewAdapter extends PagerAdapter {
    private static final String TAG = "ImagePreview";
    private AppCompatActivity activity;
    private List<ImageInfo> imageInfo;
    private HashMap<String, SubsamplingScaleImageViewDragClose> imageHashMap = new HashMap<>();
    private HashMap<String, PhotoView> imageGifHashMap = new HashMap<>();
    private String finalLoadUrl = "";

    /* renamed from: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter$7, reason: invalid class name */
    class AnonymousClass7 implements RequestListener<File> {
        final /* synthetic */ PhotoView val$imageGif;
        final /* synthetic */ SubsamplingScaleImageViewDragClose val$imageView;
        final /* synthetic */ ProgressBar val$progressBar;
        final /* synthetic */ String val$url;

        AnonymousClass7(String str, SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose, PhotoView photoView, ProgressBar progressBar) {
            this.val$url = str;
            this.val$imageView = subsamplingScaleImageViewDragClose;
            this.val$imageGif = photoView;
            this.val$progressBar = progressBar;
        }

        @Override // com.bumptech.glide.request.RequestListener
        public boolean onLoadFailed(GlideException glideException, Object obj, Target<File> target, boolean z) {
            Glide.with((FragmentActivity) ImagePreviewAdapter.this.activity).downloadOnly().load(this.val$url).addListener(new RequestListener<File>() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.7.2
                @Override // com.bumptech.glide.request.RequestListener
                public boolean onLoadFailed(GlideException glideException2, Object obj2, Target<File> target2, boolean z2) {
                    Glide.with((FragmentActivity) ImagePreviewAdapter.this.activity).downloadOnly().load(AnonymousClass7.this.val$url).addListener(new RequestListener<File>() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.7.2.2
                        @Override // com.bumptech.glide.request.RequestListener
                        public boolean onLoadFailed(GlideException glideException3, Object obj3, Target<File> target3, boolean z3) {
                            ImagePreviewAdapter.this.loadFailed(AnonymousClass7.this.val$imageView, AnonymousClass7.this.val$imageGif, AnonymousClass7.this.val$progressBar, glideException3);
                            return true;
                        }

                        @Override // com.bumptech.glide.request.RequestListener
                        public boolean onResourceReady(File file, Object obj3, Target<File> target3, DataSource dataSource, boolean z3) {
                            ImagePreviewAdapter.this.loadSuccess(file, AnonymousClass7.this.val$imageView, AnonymousClass7.this.val$imageGif, AnonymousClass7.this.val$progressBar);
                            return true;
                        }
                    }).into((RequestBuilder<File>) new FileTarget() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.7.2.1
                        @Override // com.hismith.hismithapp.image_preview.glide.FileTarget, com.bumptech.glide.request.target.Target
                        public void onLoadStarted(Drawable drawable) {
                            super.onLoadStarted(drawable);
                        }
                    });
                    return true;
                }

                @Override // com.bumptech.glide.request.RequestListener
                public boolean onResourceReady(File file, Object obj2, Target<File> target2, DataSource dataSource, boolean z2) {
                    ImagePreviewAdapter.this.loadSuccess(file, AnonymousClass7.this.val$imageView, AnonymousClass7.this.val$imageGif, AnonymousClass7.this.val$progressBar);
                    return true;
                }
            }).into((RequestBuilder<File>) new FileTarget() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.7.1
                @Override // com.hismith.hismithapp.image_preview.glide.FileTarget, com.bumptech.glide.request.target.Target
                public void onLoadStarted(Drawable drawable) {
                    super.onLoadStarted(drawable);
                }
            });
            return true;
        }

        @Override // com.bumptech.glide.request.RequestListener
        public boolean onResourceReady(File file, Object obj, Target<File> target, DataSource dataSource, boolean z) {
            ImagePreviewAdapter.this.loadSuccess(file, this.val$imageView, this.val$imageGif, this.val$progressBar);
            return true;
        }
    }

    public ImagePreviewAdapter(AppCompatActivity appCompatActivity, List<ImageInfo> list) {
        this.imageInfo = list;
        this.activity = appCompatActivity;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void loadFailed(SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose, ImageView imageView, ProgressBar progressBar, GlideException glideException) {
        progressBar.setVisibility(8);
        imageView.setVisibility(8);
        subsamplingScaleImageViewDragClose.setVisibility(0);
        subsamplingScaleImageViewDragClose.setZoomEnabled(false);
        subsamplingScaleImageViewDragClose.setImage(ImageSource.resource(ImagePreview.getInstance().getErrorPlaceHolder()));
        if (ImagePreview.getInstance().isShowErrorToast()) {
            String languageByKey = LanguageUtil.getLanguageByKey(this.activity, "failedToLoad");
            if (glideException != null) {
                languageByKey = languageByKey.concat(":\n").concat(glideException.getMessage());
            }
            if (languageByKey.length() > 200) {
                languageByKey = languageByKey.substring(0, 199);
            }
            ToastUtil.getInstance()._short(this.activity.getApplicationContext(), languageByKey);
        }
    }

    private void loadGifImageSpec(String str, final SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose, final ImageView imageView, final ProgressBar progressBar) {
        imageView.setVisibility(0);
        subsamplingScaleImageViewDragClose.setVisibility(8);
        Glide.with((FragmentActivity) this.activity).asGif().load(str).apply((BaseRequestOptions<?>) new RequestOptions().diskCacheStrategy(DiskCacheStrategy.RESOURCE).error(ImagePreview.getInstance().getErrorPlaceHolder())).listener(new RequestListener<GifDrawable>() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.9
            @Override // com.bumptech.glide.request.RequestListener
            public boolean onLoadFailed(GlideException glideException, Object obj, Target<GifDrawable> target, boolean z) {
                imageView.setVisibility(8);
                subsamplingScaleImageViewDragClose.setVisibility(0);
                subsamplingScaleImageViewDragClose.setImage(ImageSource.resource(ImagePreview.getInstance().getErrorPlaceHolder()));
                return false;
            }

            @Override // com.bumptech.glide.request.RequestListener
            public boolean onResourceReady(GifDrawable gifDrawable, Object obj, Target<GifDrawable> target, DataSource dataSource, boolean z) {
                progressBar.setVisibility(8);
                return false;
            }
        }).into(imageView);
    }

    private void loadImageSpec(String str, SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose, ImageView imageView, final ProgressBar progressBar) {
        imageView.setVisibility(8);
        subsamplingScaleImageViewDragClose.setVisibility(0);
        setImageSpec(str, subsamplingScaleImageViewDragClose);
        subsamplingScaleImageViewDragClose.setOrientation(-1);
        ImageSource uri = ImageSource.uri(Uri.fromFile(new File(str)));
        if (ImageUtil.isBmpImageWithMime(str)) {
            uri.tilingDisabled();
        }
        subsamplingScaleImageViewDragClose.setImage(uri);
        subsamplingScaleImageViewDragClose.setOnImageEventListener(new SubsamplingScaleImageViewDragClose.OnImageEventListener() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.8
            @Override // com.hismith.hismithapp.image_preview.view.helper.SubsamplingScaleImageViewDragClose.OnImageEventListener
            public void onImageLoadError(Exception exc) {
            }

            @Override // com.hismith.hismithapp.image_preview.view.helper.SubsamplingScaleImageViewDragClose.OnImageEventListener
            public void onImageLoaded() {
            }

            @Override // com.hismith.hismithapp.image_preview.view.helper.SubsamplingScaleImageViewDragClose.OnImageEventListener
            public void onPreviewLoadError(Exception exc) {
            }

            @Override // com.hismith.hismithapp.image_preview.view.helper.SubsamplingScaleImageViewDragClose.OnImageEventListener
            public void onPreviewReleased() {
            }

            @Override // com.hismith.hismithapp.image_preview.view.helper.SubsamplingScaleImageViewDragClose.OnImageEventListener
            public void onReady() {
                progressBar.setVisibility(8);
            }

            @Override // com.hismith.hismithapp.image_preview.view.helper.SubsamplingScaleImageViewDragClose.OnImageEventListener
            public void onTileLoadError(Exception exc) {
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void loadSuccess(File file, SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose, ImageView imageView, ProgressBar progressBar) {
        String absolutePath = file.getAbsolutePath();
        if (ImageUtil.isGifImageWithMime(absolutePath)) {
            loadGifImageSpec(absolutePath, subsamplingScaleImageViewDragClose, imageView, progressBar);
        } else {
            loadImageSpec(absolutePath, subsamplingScaleImageViewDragClose, imageView, progressBar);
        }
    }

    private void setImageSpec(String str, SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose) {
        if (ImageUtil.isLongImage(this.activity, str)) {
            subsamplingScaleImageViewDragClose.setMinimumScaleType(4);
            subsamplingScaleImageViewDragClose.setMinScale(ImageUtil.getLongImageMinScale(this.activity, str));
            subsamplingScaleImageViewDragClose.setMaxScale(ImageUtil.getLongImageMaxScale(this.activity, str));
            subsamplingScaleImageViewDragClose.setDoubleTapZoomScale(ImageUtil.getLongImageMaxScale(this.activity, str));
            return;
        }
        boolean isWideImage = ImageUtil.isWideImage(this.activity, str);
        boolean isSmallImage = ImageUtil.isSmallImage(this.activity, str);
        if (isWideImage) {
            subsamplingScaleImageViewDragClose.setMinimumScaleType(1);
            subsamplingScaleImageViewDragClose.setMinScale(ImagePreview.getInstance().getMinScale());
            subsamplingScaleImageViewDragClose.setMaxScale(ImagePreview.getInstance().getMaxScale());
            subsamplingScaleImageViewDragClose.setDoubleTapZoomScale(ImageUtil.getWideImageDoubleScale(this.activity, str));
            return;
        }
        if (isSmallImage) {
            subsamplingScaleImageViewDragClose.setMinimumScaleType(3);
            subsamplingScaleImageViewDragClose.setMinScale(ImageUtil.getSmallImageMinScale(this.activity, str));
            subsamplingScaleImageViewDragClose.setMaxScale(ImageUtil.getSmallImageMaxScale(this.activity, str));
            subsamplingScaleImageViewDragClose.setDoubleTapZoomScale(ImageUtil.getSmallImageMaxScale(this.activity, str));
            return;
        }
        subsamplingScaleImageViewDragClose.setMinimumScaleType(1);
        subsamplingScaleImageViewDragClose.setMinScale(ImagePreview.getInstance().getMinScale());
        subsamplingScaleImageViewDragClose.setMaxScale(ImagePreview.getInstance().getMaxScale());
        subsamplingScaleImageViewDragClose.setDoubleTapZoomScale(ImagePreview.getInstance().getMediumScale());
    }

    public void closePage() {
        try {
            HashMap<String, SubsamplingScaleImageViewDragClose> hashMap = this.imageHashMap;
            if (hashMap != null && hashMap.size() > 0) {
                for (Map.Entry<String, SubsamplingScaleImageViewDragClose> entry : this.imageHashMap.entrySet()) {
                    if (entry != null && entry.getValue() != null) {
                        entry.getValue().destroyDrawingCache();
                        entry.getValue().recycle();
                    }
                }
                this.imageHashMap.clear();
                this.imageHashMap = null;
            }
            HashMap<String, PhotoView> hashMap2 = this.imageGifHashMap;
            if (hashMap2 == null || hashMap2.size() <= 0) {
                return;
            }
            for (Map.Entry<String, PhotoView> entry2 : this.imageGifHashMap.entrySet()) {
                if (entry2 != null && entry2.getValue() != null) {
                    entry2.getValue().destroyDrawingCache();
                    entry2.getValue().setImageBitmap(null);
                }
            }
            this.imageGifHashMap.clear();
            this.imageGifHashMap = null;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public void destroyItem(ViewGroup viewGroup, int i, Object obj) {
        PhotoView photoView;
        SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose;
        String originUrl = this.imageInfo.get(i).getOriginUrl();
        try {
            HashMap<String, SubsamplingScaleImageViewDragClose> hashMap = this.imageHashMap;
            if (hashMap != null && (subsamplingScaleImageViewDragClose = hashMap.get(originUrl)) != null) {
                subsamplingScaleImageViewDragClose.resetScaleAndCenter();
                subsamplingScaleImageViewDragClose.destroyDrawingCache();
                subsamplingScaleImageViewDragClose.recycle();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            HashMap<String, PhotoView> hashMap2 = this.imageGifHashMap;
            if (hashMap2 != null && (photoView = hashMap2.get(originUrl)) != null) {
                photoView.destroyDrawingCache();
                photoView.setImageBitmap(null);
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        try {
            ImageLoader.clearMemory(this.activity);
        } catch (Exception e3) {
            e3.printStackTrace();
        }
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public int getCount() {
        return this.imageInfo.size();
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public int getItemPosition(Object obj) {
        return -2;
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public Object instantiateItem(ViewGroup viewGroup, final int i) {
        AppCompatActivity appCompatActivity = this.activity;
        if (appCompatActivity == null) {
            return viewGroup;
        }
        View inflate = View.inflate(appCompatActivity, R.layout.sh_item_photoview, null);
        ProgressBar progressBar = (ProgressBar) inflate.findViewById(R.id.progress_view);
        FingerDragHelper fingerDragHelper = (FingerDragHelper) inflate.findViewById(R.id.fingerDragHelper);
        final SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose = (SubsamplingScaleImageViewDragClose) inflate.findViewById(R.id.photo_view);
        final PhotoView photoView = (PhotoView) inflate.findViewById(R.id.gif_view);
        ImageInfo imageInfo = this.imageInfo.get(i);
        String originUrl = imageInfo.getOriginUrl();
        String thumbnailUrl = imageInfo.getThumbnailUrl();
        subsamplingScaleImageViewDragClose.setMinimumScaleType(1);
        subsamplingScaleImageViewDragClose.setDoubleTapZoomStyle(2);
        subsamplingScaleImageViewDragClose.setDoubleTapZoomDuration(ImagePreview.getInstance().getZoomTransitionDuration());
        subsamplingScaleImageViewDragClose.setMinScale(ImagePreview.getInstance().getMinScale());
        subsamplingScaleImageViewDragClose.setMaxScale(ImagePreview.getInstance().getMaxScale());
        subsamplingScaleImageViewDragClose.setDoubleTapZoomScale(ImagePreview.getInstance().getMediumScale());
        photoView.setZoomTransitionDuration(ImagePreview.getInstance().getZoomTransitionDuration());
        photoView.setMinimumScale(ImagePreview.getInstance().getMinScale());
        photoView.setMaximumScale(ImagePreview.getInstance().getMaxScale());
        photoView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        subsamplingScaleImageViewDragClose.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (ImagePreview.getInstance().isEnableClickClose()) {
                    ImagePreviewAdapter.this.activity.finish();
                }
                if (ImagePreview.getInstance().getBigImageClickListener() != null) {
                    ImagePreview.getInstance().getBigImageClickListener().onClick(ImagePreviewAdapter.this.activity, view, i);
                }
            }
        });
        photoView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (ImagePreview.getInstance().isEnableClickClose()) {
                    ImagePreviewAdapter.this.activity.finish();
                }
                if (ImagePreview.getInstance().getBigImageClickListener() != null) {
                    ImagePreview.getInstance().getBigImageClickListener().onClick(ImagePreviewAdapter.this.activity, view, i);
                }
            }
        });
        subsamplingScaleImageViewDragClose.setOnLongClickListener(new View.OnLongClickListener() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.3
            @Override // android.view.View.OnLongClickListener
            public boolean onLongClick(View view) {
                if (ImagePreview.getInstance().getBigImageLongClickListener() != null) {
                    return ImagePreview.getInstance().getBigImageLongClickListener().onLongClick(ImagePreviewAdapter.this.activity, view, i);
                }
                return false;
            }
        });
        photoView.setOnLongClickListener(new View.OnLongClickListener() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.4
            @Override // android.view.View.OnLongClickListener
            public boolean onLongClick(View view) {
                if (ImagePreview.getInstance().getBigImageLongClickListener() != null) {
                    return ImagePreview.getInstance().getBigImageLongClickListener().onLongClick(ImagePreviewAdapter.this.activity, view, i);
                }
                return false;
            }
        });
        if (ImagePreview.getInstance().isEnableDragClose()) {
            fingerDragHelper.setOnAlphaChangeListener(new FingerDragHelper.onAlphaChangedListener() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.5
                @Override // com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper.onAlphaChangedListener
                public void onTranslationYChanged(MotionEvent motionEvent, float f) {
                    float abs = 1.0f - (Math.abs(f) / PhoneUtil.getPhoneHei(ImagePreviewAdapter.this.activity.getApplicationContext()));
                    if (ImagePreviewAdapter.this.activity instanceof ImagePreviewActivity) {
                        ((ImagePreviewActivity) ImagePreviewAdapter.this.activity).setAlpha(abs);
                    }
                    if (photoView.getVisibility() == 0) {
                        photoView.setScaleY(abs);
                        photoView.setScaleX(abs);
                    }
                    if (subsamplingScaleImageViewDragClose.getVisibility() == 0) {
                        subsamplingScaleImageViewDragClose.setScaleY(abs);
                        subsamplingScaleImageViewDragClose.setScaleX(abs);
                    }
                }
            });
        }
        this.imageGifHashMap.remove(originUrl);
        this.imageGifHashMap.put(originUrl, photoView);
        this.imageHashMap.remove(originUrl);
        this.imageHashMap.put(originUrl, subsamplingScaleImageViewDragClose);
        ImagePreview.LoadStrategy loadStrategy = ImagePreview.getInstance().getLoadStrategy();
        if (loadStrategy == ImagePreview.LoadStrategy.Default) {
            this.finalLoadUrl = thumbnailUrl;
        } else if (loadStrategy == ImagePreview.LoadStrategy.AlwaysOrigin) {
            this.finalLoadUrl = originUrl;
        } else if (loadStrategy == ImagePreview.LoadStrategy.AlwaysThumb) {
            this.finalLoadUrl = thumbnailUrl;
        } else if (loadStrategy == ImagePreview.LoadStrategy.NetworkAuto) {
            if (NetworkUtil.isWiFi(this.activity)) {
                this.finalLoadUrl = originUrl;
            } else {
                this.finalLoadUrl = thumbnailUrl;
            }
        }
        String trim = this.finalLoadUrl.trim();
        this.finalLoadUrl = trim;
        progressBar.setVisibility(0);
        File glideCacheFile = ImageLoader.getGlideCacheFile(this.activity, originUrl);
        if (glideCacheFile == null || !glideCacheFile.exists()) {
            Glide.with((FragmentActivity) this.activity).downloadOnly().load(trim).addListener(new AnonymousClass7(trim, subsamplingScaleImageViewDragClose, photoView, progressBar)).into((RequestBuilder<File>) new FileTarget() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewAdapter.6
                @Override // com.hismith.hismithapp.image_preview.glide.FileTarget, com.bumptech.glide.request.target.Target
                public void onLoadStarted(Drawable drawable) {
                    super.onLoadStarted(drawable);
                }
            });
        } else if (ImageUtil.isGifImageWithMime(glideCacheFile.getAbsolutePath())) {
            loadGifImageSpec(glideCacheFile.getAbsolutePath(), subsamplingScaleImageViewDragClose, photoView, progressBar);
        } else {
            loadImageSpec(glideCacheFile.getAbsolutePath(), subsamplingScaleImageViewDragClose, photoView, progressBar);
        }
        viewGroup.addView(inflate);
        return inflate;
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public boolean isViewFromObject(View view, Object obj) {
        return view == obj;
    }

    public void loadOrigin(ImageInfo imageInfo) {
        ImageSource imageSource;
        String originUrl = imageInfo.getOriginUrl();
        HashMap<String, SubsamplingScaleImageViewDragClose> hashMap = this.imageHashMap;
        if (hashMap == null || this.imageGifHashMap == null) {
            notifyDataSetChanged();
            return;
        }
        if (hashMap.get(originUrl) == null || this.imageGifHashMap.get(originUrl) == null) {
            notifyDataSetChanged();
            return;
        }
        SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose = this.imageHashMap.get(imageInfo.getOriginUrl());
        PhotoView photoView = this.imageGifHashMap.get(imageInfo.getOriginUrl());
        File glideCacheFile = ImageLoader.getGlideCacheFile(this.activity, imageInfo.getOriginUrl());
        if (glideCacheFile == null || !glideCacheFile.exists()) {
            notifyDataSetChanged();
            return;
        }
        if (ImageUtil.isGifImageWithMime(glideCacheFile.getAbsolutePath())) {
            if (subsamplingScaleImageViewDragClose != null) {
                subsamplingScaleImageViewDragClose.setVisibility(8);
            }
            if (photoView != null) {
                photoView.setVisibility(0);
                Glide.with((FragmentActivity) this.activity).asGif().load(glideCacheFile).apply((BaseRequestOptions<?>) new RequestOptions().diskCacheStrategy(DiskCacheStrategy.RESOURCE).error(ImagePreview.getInstance().getErrorPlaceHolder())).into(photoView);
                return;
            }
            return;
        }
        if (photoView != null) {
            photoView.setVisibility(8);
        }
        if (subsamplingScaleImageViewDragClose != null) {
            subsamplingScaleImageViewDragClose.setVisibility(0);
            File glideCacheFile2 = ImageLoader.getGlideCacheFile(this.activity, imageInfo.getThumbnailUrl());
            if (glideCacheFile2 == null || !glideCacheFile2.exists()) {
                imageSource = null;
            } else {
                String absolutePath = glideCacheFile2.getAbsolutePath();
                imageSource = ImageSource.bitmap(ImageUtil.getImageBitmap(absolutePath, ImageUtil.getBitmapDegree(absolutePath)));
                int i = ImageUtil.getWidthHeight(absolutePath)[0];
                int i2 = ImageUtil.getWidthHeight(absolutePath)[1];
                if (ImageUtil.isBmpImageWithMime(glideCacheFile.getAbsolutePath())) {
                    imageSource.tilingDisabled();
                }
                imageSource.dimensions(i, i2);
            }
            String absolutePath2 = glideCacheFile.getAbsolutePath();
            ImageSource uri = ImageSource.uri(absolutePath2);
            int i3 = ImageUtil.getWidthHeight(absolutePath2)[0];
            int i4 = ImageUtil.getWidthHeight(absolutePath2)[1];
            if (ImageUtil.isBmpImageWithMime(glideCacheFile.getAbsolutePath())) {
                uri.tilingDisabled();
            }
            uri.dimensions(i3, i4);
            setImageSpec(absolutePath2, subsamplingScaleImageViewDragClose);
            subsamplingScaleImageViewDragClose.setOrientation(-1);
            subsamplingScaleImageViewDragClose.setImage(uri, imageSource);
        }
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public void setPrimaryItem(ViewGroup viewGroup, int i, Object obj) {
        super.setPrimaryItem(viewGroup, i, obj);
    }
}
