package com.hismith.hismithapp.image_preview.view;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.viewpager.widget.ViewPager;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.image_preview.bean.ImageInfo;
import com.hismith.hismithapp.image_preview.glide.FileTarget;
import com.hismith.hismithapp.image_preview.glide.ImageLoader;
import com.hismith.hismithapp.image_preview.glide.progress.OnProgressListener;
import com.hismith.hismithapp.image_preview.glide.progress.ProgressManager;
import com.hismith.hismithapp.image_preview.tool.common.HandlerUtils;
import com.hismith.hismithapp.image_preview.tool.image.DownloadPictureUtil;
import com.hismith.hismithapp.image_preview.tool.ui.ToastUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.luck.picture.lib.permissions.PermissionConfig;
import java.io.File;
import java.util.List;

/* loaded from: classes3.dex */
public class ImagePreviewActivity extends AppCompatActivity implements Handler.Callback, View.OnClickListener {
    public static final String TAG = "ImagePreview";
    private Button btn_show_origin;
    private Context context;
    private int currentItem;
    private FrameLayout fm_center_progress_container;
    private FrameLayout fm_image_show_origin_container;
    private HandlerUtils.HandlerHolder handlerHolder;
    private List<ImageInfo> imageInfoList;
    private ImagePreviewAdapter imagePreviewAdapter;
    private ImageView imgCloseButton;
    private ImageView img_download;
    private boolean isShowCloseButton;
    private boolean isShowDownButton;
    private boolean isShowIndicator;
    private boolean isShowOriginButton;
    private View progressParentLayout;
    private View rootView;
    private TextView tv_id;
    private TextView tv_indicator;
    private HackyViewPager viewPager;
    private boolean isUserCustomProgressView = false;
    private boolean indicatorStatus = false;
    private boolean originalStatus = false;
    private boolean downloadButtonStatus = false;
    private boolean closeButtonStatus = false;
    private String currentItemOriginPathUrl = "";
    private int lastProgress = 0;

    public static void activityStart(Context context) {
        if (context == null) {
            return;
        }
        Intent intent = new Intent();
        intent.setClass(context, ImagePreviewActivity.class);
        context.startActivity(intent);
        if (context instanceof Activity) {
            ((Activity) context).overridePendingTransition(R.anim.fade_in, R.anim.fade_out);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean checkCache(String str) {
        File glideCacheFile = ImageLoader.getGlideCacheFile(this.context, str);
        if (glideCacheFile == null || !glideCacheFile.exists()) {
            visible();
            return false;
        }
        gone();
        return true;
    }

    private void downloadCurrentImg() {
        DownloadPictureUtil.downloadPicture(this.context.getApplicationContext(), this.currentItemOriginPathUrl);
    }

    private int getRealIndexWithPath(String str) {
        for (int i = 0; i < this.imageInfoList.size(); i++) {
            if (str.equalsIgnoreCase(this.imageInfoList.get(i).getOriginUrl())) {
                return i;
            }
        }
        return 0;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void gone() {
        this.handlerHolder.sendEmptyMessage(3);
    }

    private void loadOriginImage(String str) {
        Glide.with(this.context).downloadOnly().load(str).into((RequestBuilder<File>) new FileTarget() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewActivity.3
            /* JADX WARN: Can't rename method to resolve collision */
            @Override // com.hismith.hismithapp.image_preview.glide.FileTarget, com.bumptech.glide.request.target.Target
            public void onResourceReady(File file, Transition<? super File> transition) {
                super.onResourceReady(file, transition);
            }
        });
        ProgressManager.addListener(str, new OnProgressListener() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewActivity.4
            @Override // com.hismith.hismithapp.image_preview.glide.progress.OnProgressListener
            public void onProgress(String str2, boolean z, int i, long j, long j2) {
                ImagePreviewActivity imagePreviewActivity = ImagePreviewActivity.this;
                if (z) {
                    Message obtainMessage = imagePreviewActivity.handlerHolder.obtainMessage();
                    Bundle bundle = new Bundle();
                    bundle.putString("url", str2);
                    obtainMessage.what = 1;
                    obtainMessage.obj = bundle;
                    ImagePreviewActivity.this.handlerHolder.sendMessage(obtainMessage);
                    return;
                }
                if (i == imagePreviewActivity.lastProgress) {
                    return;
                }
                ImagePreviewActivity.this.lastProgress = i;
                Message obtainMessage2 = ImagePreviewActivity.this.handlerHolder.obtainMessage();
                Bundle bundle2 = new Bundle();
                bundle2.putString("url", str2);
                bundle2.putInt("progress", i);
                obtainMessage2.what = 2;
                obtainMessage2.obj = bundle2;
                ImagePreviewActivity.this.handlerHolder.sendMessage(obtainMessage2);
            }
        });
    }

    private void visible() {
        this.handlerHolder.sendEmptyMessage(4);
    }

    public int convertPercentToBlackAlphaColor(float f) {
        String lowerCase = Integer.toHexString((int) (Math.min(1.0f, Math.max(0.0f, f)) * 255.0f)).toLowerCase();
        return Color.parseColor("#" + (lowerCase.length() < 2 ? "0" : "") + lowerCase + "000000");
    }

    @Override // android.app.Activity
    public void finish() {
        super.finish();
        overridePendingTransition(R.anim.fade_in, R.anim.fade_out);
    }

    @Override // android.os.Handler.Callback
    public boolean handleMessage(Message message) {
        if (message.what == 0) {
            String originUrl = this.imageInfoList.get(this.currentItem).getOriginUrl();
            visible();
            if (this.isUserCustomProgressView) {
                gone();
            } else {
                this.btn_show_origin.setText("0 %");
            }
            if (checkCache(originUrl)) {
                Message obtainMessage = this.handlerHolder.obtainMessage();
                Bundle bundle = new Bundle();
                bundle.putString("url", originUrl);
                obtainMessage.what = 1;
                obtainMessage.obj = bundle;
                this.handlerHolder.sendMessage(obtainMessage);
                return true;
            }
            loadOriginImage(originUrl);
        } else if (message.what == 1) {
            String string = ((Bundle) message.obj).getString("url");
            gone();
            if (this.currentItem == getRealIndexWithPath(string)) {
                if (this.isUserCustomProgressView) {
                    this.fm_center_progress_container.setVisibility(8);
                    if (ImagePreview.getInstance().getOnOriginProgressListener() != null) {
                        this.progressParentLayout.setVisibility(8);
                        ImagePreview.getInstance().getOnOriginProgressListener().finish(this.progressParentLayout);
                    }
                    this.imagePreviewAdapter.loadOrigin(this.imageInfoList.get(this.currentItem));
                } else {
                    this.imagePreviewAdapter.loadOrigin(this.imageInfoList.get(this.currentItem));
                }
            }
        } else if (message.what == 2) {
            Bundle bundle2 = (Bundle) message.obj;
            String string2 = bundle2.getString("url");
            int i = bundle2.getInt("progress");
            if (this.currentItem == getRealIndexWithPath(string2)) {
                if (this.isUserCustomProgressView) {
                    gone();
                    this.fm_center_progress_container.setVisibility(0);
                    if (ImagePreview.getInstance().getOnOriginProgressListener() != null) {
                        this.progressParentLayout.setVisibility(0);
                        ImagePreview.getInstance().getOnOriginProgressListener().progress(this.progressParentLayout, i);
                    }
                } else {
                    visible();
                    this.btn_show_origin.setText(String.format("%s %%", String.valueOf(i)));
                }
            }
        } else if (message.what == 3) {
            LanguageUtil.setText(this.context, this.btn_show_origin, "viewOriginalPhoto");
            this.fm_image_show_origin_container.setVisibility(8);
            this.originalStatus = false;
        } else if (message.what == 4) {
            this.fm_image_show_origin_container.setVisibility(0);
            this.originalStatus = true;
        }
        return true;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id != R.id.img_download) {
            if (id == R.id.btn_show_origin) {
                this.handlerHolder.sendEmptyMessage(0);
                return;
            } else {
                if (id == R.id.imgCloseButton) {
                    onBackPressed();
                    return;
                }
                return;
            }
        }
        if (ContextCompat.checkSelfPermission(this.context, PermissionConfig.WRITE_EXTERNAL_STORAGE) == 0 || Build.VERSION.SDK_INT >= 33) {
            downloadCurrentImg();
        } else if (ActivityCompat.shouldShowRequestPermissionRationale(this, PermissionConfig.WRITE_EXTERNAL_STORAGE)) {
            ToastUtil.getInstance()._short(this.context, LanguageUtil.getLanguageByKey(this.context, "permissionDenied"));
        } else {
            ActivityCompat.requestPermissions(this, new String[]{PermissionConfig.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewActivity.1
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                ImagePreviewActivity.this.finish();
            }
        });
        setContentView(R.layout.sh_layout_preview);
        Window window = getWindow();
        window.clearFlags(67108864);
        window.getDecorView().setSystemUiVisibility(1280);
        window.addFlags(Integer.MIN_VALUE);
        window.setStatusBarColor(0);
        this.context = this;
        this.handlerHolder = new HandlerUtils.HandlerHolder(this);
        List<ImageInfo> imageInfoList = ImagePreview.getInstance().getImageInfoList();
        this.imageInfoList = imageInfoList;
        if (imageInfoList == null || imageInfoList.size() == 0) {
            onBackPressed();
            return;
        }
        this.currentItem = ImagePreview.getInstance().getIndex();
        this.isShowDownButton = ImagePreview.getInstance().isShowDownButton();
        this.isShowCloseButton = ImagePreview.getInstance().isShowCloseButton();
        this.isShowIndicator = ImagePreview.getInstance().isShowIndicator();
        this.currentItemOriginPathUrl = this.imageInfoList.get(this.currentItem).getOriginUrl();
        boolean isShowOriginButton = ImagePreview.getInstance().isShowOriginButton(this.currentItem);
        this.isShowOriginButton = isShowOriginButton;
        if (isShowOriginButton) {
            checkCache(this.currentItemOriginPathUrl);
        }
        this.rootView = findViewById(R.id.rootView);
        this.viewPager = (HackyViewPager) findViewById(R.id.viewPager);
        this.tv_id = (TextView) findViewById(R.id.tv_id);
        this.tv_indicator = (TextView) findViewById(R.id.tv_indicator);
        this.fm_image_show_origin_container = (FrameLayout) findViewById(R.id.fm_image_show_origin_container);
        this.fm_center_progress_container = (FrameLayout) findViewById(R.id.fm_center_progress_container);
        this.fm_image_show_origin_container.setVisibility(8);
        this.fm_center_progress_container.setVisibility(8);
        if (ImagePreview.getInstance().getProgressLayoutId() != -1) {
            View inflate = View.inflate(this.context, ImagePreview.getInstance().getProgressLayoutId(), null);
            this.progressParentLayout = inflate;
            if (inflate != null) {
                this.fm_center_progress_container.removeAllViews();
                this.fm_center_progress_container.addView(this.progressParentLayout);
                this.isUserCustomProgressView = true;
            } else {
                this.isUserCustomProgressView = false;
            }
        } else {
            this.isUserCustomProgressView = false;
        }
        this.btn_show_origin = (Button) findViewById(R.id.btn_show_origin);
        this.img_download = (ImageView) findViewById(R.id.img_download);
        this.imgCloseButton = (ImageView) findViewById(R.id.imgCloseButton);
        LanguageUtil.setText(this.context, this.btn_show_origin, "viewOriginalPhoto");
        this.img_download.setImageResource(ImagePreview.getInstance().getDownIconResId());
        this.imgCloseButton.setImageResource(ImagePreview.getInstance().getCloseIconResId());
        this.imgCloseButton.setOnClickListener(this);
        this.btn_show_origin.setOnClickListener(this);
        this.img_download.setOnClickListener(this);
        if (ImagePreview.getInstance().getUserId() > 0) {
            this.tv_id.setVisibility(0);
            this.tv_id.setText(String.format("u:%1$s", Integer.valueOf(ImagePreview.getInstance().getUserId())));
        } else {
            int topicId = ImagePreview.getInstance().getTopicId();
            TextView textView = this.tv_id;
            if (topicId > 0) {
                textView.setVisibility(0);
                this.tv_id.setText(String.format("t:%1$s", Integer.valueOf(ImagePreview.getInstance().getTopicId())));
            } else {
                textView.setVisibility(8);
            }
        }
        if (this.isShowIndicator) {
            int size = this.imageInfoList.size();
            TextView textView2 = this.tv_indicator;
            if (size > 1) {
                textView2.setVisibility(0);
                this.indicatorStatus = true;
            } else {
                textView2.setVisibility(8);
                this.indicatorStatus = false;
            }
        } else {
            this.tv_indicator.setVisibility(8);
            this.indicatorStatus = false;
        }
        if (ImagePreview.getInstance().getIndicatorShapeResId() > 0) {
            this.tv_indicator.setBackgroundResource(ImagePreview.getInstance().getIndicatorShapeResId());
        }
        boolean z = this.isShowDownButton;
        ImageView imageView = this.img_download;
        if (z) {
            imageView.setVisibility(0);
            this.downloadButtonStatus = true;
        } else {
            imageView.setVisibility(8);
            this.downloadButtonStatus = false;
        }
        boolean z2 = this.isShowCloseButton;
        ImageView imageView2 = this.imgCloseButton;
        if (z2) {
            imageView2.setVisibility(0);
            this.closeButtonStatus = true;
        } else {
            imageView2.setVisibility(8);
            this.closeButtonStatus = false;
        }
        this.tv_indicator.setText(String.format(getString(R.string.indicator), (this.currentItem + 1) + "", "" + this.imageInfoList.size()));
        ImagePreviewAdapter imagePreviewAdapter = new ImagePreviewAdapter(this, this.imageInfoList);
        this.imagePreviewAdapter = imagePreviewAdapter;
        this.viewPager.setAdapter(imagePreviewAdapter);
        this.viewPager.setCurrentItem(this.currentItem);
        this.viewPager.addOnPageChangeListener(new ViewPager.SimpleOnPageChangeListener() { // from class: com.hismith.hismithapp.image_preview.view.ImagePreviewActivity.2
            @Override // androidx.viewpager.widget.ViewPager.SimpleOnPageChangeListener, androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrollStateChanged(int i) {
                super.onPageScrollStateChanged(i);
                if (ImagePreview.getInstance().getBigImagePageChangeListener() != null) {
                    ImagePreview.getInstance().getBigImagePageChangeListener().onPageScrollStateChanged(i);
                }
            }

            @Override // androidx.viewpager.widget.ViewPager.SimpleOnPageChangeListener, androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrolled(int i, float f, int i2) {
                super.onPageScrolled(i, f, i2);
                if (ImagePreview.getInstance().getBigImagePageChangeListener() != null) {
                    ImagePreview.getInstance().getBigImagePageChangeListener().onPageScrolled(i, f, i2);
                }
            }

            @Override // androidx.viewpager.widget.ViewPager.SimpleOnPageChangeListener, androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageSelected(int i) {
                super.onPageSelected(i);
                if (ImagePreview.getInstance().getBigImagePageChangeListener() != null) {
                    ImagePreview.getInstance().getBigImagePageChangeListener().onPageSelected(i);
                }
                ImagePreviewActivity.this.currentItem = i;
                ImagePreviewActivity imagePreviewActivity = ImagePreviewActivity.this;
                imagePreviewActivity.currentItemOriginPathUrl = ((ImageInfo) imagePreviewActivity.imageInfoList.get(i)).getOriginUrl();
                ImagePreviewActivity.this.isShowOriginButton = ImagePreview.getInstance().isShowOriginButton(ImagePreviewActivity.this.currentItem);
                boolean z3 = ImagePreviewActivity.this.isShowOriginButton;
                ImagePreviewActivity imagePreviewActivity2 = ImagePreviewActivity.this;
                if (z3) {
                    imagePreviewActivity2.checkCache(imagePreviewActivity2.currentItemOriginPathUrl);
                } else {
                    imagePreviewActivity2.gone();
                }
                ImagePreviewActivity.this.tv_indicator.setText(String.format(ImagePreviewActivity.this.getString(R.string.indicator), (ImagePreviewActivity.this.currentItem + 1) + "", "" + ImagePreviewActivity.this.imageInfoList.size()));
                if (ImagePreviewActivity.this.isUserCustomProgressView) {
                    ImagePreviewActivity.this.fm_center_progress_container.setVisibility(8);
                    ImagePreviewActivity.this.lastProgress = 0;
                }
            }
        });
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        ImagePreview.getInstance().reset();
        ImagePreviewAdapter imagePreviewAdapter = this.imagePreviewAdapter;
        if (imagePreviewAdapter != null) {
            imagePreviewAdapter.closePage();
        }
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
        if (i == 1) {
            for (int i2 = 0; i2 < strArr.length; i2++) {
                if (iArr[i2] == 0) {
                    downloadCurrentImg();
                } else {
                    ToastUtil.getInstance()._short(this.context, LanguageUtil.getLanguageByKey(this.context, "permissionDenied"));
                }
            }
        }
    }

    public void setAlpha(float f) {
        this.rootView.setBackgroundColor(convertPercentToBlackAlphaColor(f));
        if (f < 1.0f) {
            this.tv_id.setVisibility(8);
            this.tv_indicator.setVisibility(8);
            this.fm_image_show_origin_container.setVisibility(8);
            this.img_download.setVisibility(8);
            this.imgCloseButton.setVisibility(8);
            return;
        }
        if (ImagePreview.getInstance().getUserId() > 0 || ImagePreview.getInstance().getTopicId() > 0) {
            this.tv_id.setVisibility(0);
        }
        if (this.indicatorStatus) {
            this.tv_indicator.setVisibility(0);
        }
        if (this.originalStatus) {
            this.fm_image_show_origin_container.setVisibility(0);
        }
        if (this.downloadButtonStatus) {
            this.img_download.setVisibility(0);
        }
        if (this.closeButtonStatus) {
            this.imgCloseButton.setVisibility(0);
        }
    }
}
