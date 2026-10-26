package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager.widget.PagerAdapter;
import androidx.viewpager.widget.ViewPager;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.RequestManager;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.facebook.appevents.UserDataStore;
import com.facebook.internal.NativeProtocol;
import com.google.android.material.appbar.AppBarLayout;
import com.google.android.material.tabs.TabLayout;
import com.google.gson.Gson;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.adapter.UserPhotosPagerAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ChoosePhotoDialog;
import com.hismith.hismithapp.fragment.PersonalMModeFragment;
import com.hismith.hismithapp.fragment.PersonalMTopicFragment;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.manager.PictureResultManager;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.model.UserPhotosModel;
import com.hismith.hismithapp.picker.PictureSelectorManager;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.RequestBodyUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StatusBarUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.luck.picture.lib.config.PictureMimeType;
import com.luck.picture.lib.entity.LocalMedia;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshHeader;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshListener;
import com.scwang.smart.refresh.layout.simple.SimpleMultiListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import okhttp3.RequestBody;

/* loaded from: classes3.dex */
public class PersonalDetailsActivity2 extends BaseActivity implements View.OnClickListener, ViewPager.OnPageChangeListener, MediaPlayer.OnPreparedListener, MediaPlayer.OnCompletionListener, MediaPlayer.OnErrorListener {
    private RetrofitRequest baseRequest;
    private RetrofitRequest fileRequest;
    private List<Fragment> fragments;
    private Gson gson;
    private List<String> imageList;
    private AppBarLayout mAppbarLayout;
    private ImageView mDetailsImgGender;
    private TextView mDetailsTxtBios;
    private TextView mDetailsTxtConstellation;
    private TextView mDetailsTxtFans;
    private TextView mDetailsTxtFansLabel;
    private TextView mDetailsTxtFocus;
    private TextView mDetailsTxtFocusLabel;
    private TextView mDetailsTxtId;
    private TextView mDetailsTxtIdLabel;
    private TextView mDetailsTxtLocation;
    private TextView mDetailsTxtName;
    private FragmentAdapter mFragmentAdapter;
    private ViewPager mFragmentViewPager;
    private CircleImageView mImgHead;
    private ImageView mImgVoice;
    private ImageView mImgVoiceB;
    private LinearLayout mLayoutChat;
    private LinearLayout mLayoutDots;
    private LinearLayout mLayoutEdit;
    private LinearLayout mLayoutFollow;
    private LinearLayout mLayoutInterestedIn;
    private RelativeLayout mLayoutPhoto;
    private FrameLayout mLayoutRoot;
    private LinearLayout mLayoutVoice;
    private PersonalMModeFragment mPersonalMModeFragment;
    private PersonalMTopicFragment mPersonalMTopicFragment;
    private SmartRefreshLayout mRefreshLayout;
    private TabLayout mTabs;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtEdit;
    private Toolbar mToolbar;
    private Toolbar mToolbarPlaceholder;
    private TextView mTxtChat;
    private TextView mTxtEdit;
    private TextView mTxtFollow;
    private TextView mTxtInterestedIn;
    private TextView mTxtInterestedInLabel;
    private TextView mTxtLocationLabel;
    private TextView mTxtVoiceTime;
    private ViewPager mViewPager;
    private MediaPlayer mediaPlayer;
    private List<View> pagerView;
    private UserPhotosModel[] photos;
    private PictureResultManager pictureResultManager;
    private final int firstAlpha = 0;
    private int isVoiceStauts = 0;
    PagerAdapter mPagerAdapter = new PagerAdapter() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.8
        @Override // androidx.viewpager.widget.PagerAdapter
        public void destroyItem(View view, int i, Object obj) {
            if (i < PersonalDetailsActivity2.this.pagerView.size()) {
                ((ViewPager) view).removeView((View) PersonalDetailsActivity2.this.pagerView.get(i));
            }
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public int getCount() {
            return PersonalDetailsActivity2.this.pagerView.size();
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public Object instantiateItem(View view, final int i) {
            ImageView imageView = (ImageView) ((View) PersonalDetailsActivity2.this.pagerView.get(i)).findViewById(R.id.item_details_image);
            imageView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.8.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view2) {
                    if (PersonalDetailsActivity2.this.imageList == null || PersonalDetailsActivity2.this.imageList.size() <= i) {
                        return;
                    }
                    ImagePreview.getInstance().setContext(PersonalDetailsActivity2.this).setIndex(i).setImageList(PersonalDetailsActivity2.this.imageList).setFolderName("hismith/image").setEnableDragClose(true).start();
                }
            });
            Glide.with((FragmentActivity) PersonalDetailsActivity2.this).load(PersonalDetailsActivity2.this.photos[i].getUrl()).into(imageView);
            ((ViewPager) view).addView((View) PersonalDetailsActivity2.this.pagerView.get(i));
            return PersonalDetailsActivity2.this.pagerView.get(i);
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public boolean isViewFromObject(View view, Object obj) {
            return view == obj;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.13
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (intent.getAction().equals(Constans.RECEIVER_USER_INFO_CHANGE)) {
                PersonalDetailsActivity2.this.initUserInfo(MyApplication.getApplication().getmUserModel());
            }
        }
    };

    /* JADX INFO: Access modifiers changed from: private */
    public void getMyInfo() {
        this.baseRequest.getApi().getMyInfo((String) SharedPreferencesUtil.get(this, Constans.Token), MyApplication.FCM_TOKEN, AppUtil.getAppVersionName(this)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.10
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                if (PersonalDetailsActivity2.this.mRefreshLayout != null) {
                    PersonalDetailsActivity2.this.mRefreshLayout.finishRefresh();
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                if (PersonalDetailsActivity2.this.mRefreshLayout != null) {
                    PersonalDetailsActivity2.this.mRefreshLayout.finishRefresh();
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel.getSuccess() == 1) {
                    MyApplication.getApplication().setmUserModel(userModel);
                    PersonalDetailsActivity2.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.fileRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_FILE);
        this.gson = new Gson();
        this.mRefreshLayout.setEnableRefresh(true);
        this.mRefreshLayout.setEnableLoadMore(false);
        ArrayList arrayList = new ArrayList();
        arrayList.add(LanguageUtil.getLanguageByKey(this, "topicTitle"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "mode"));
        for (int i = 0; i < arrayList.size(); i++) {
            TabLayout.Tab newTab = this.mTabs.newTab();
            newTab.setText((CharSequence) arrayList.get(i));
            newTab.setTag(Integer.valueOf(i));
            this.mTabs.addTab(newTab);
        }
        ArrayList arrayList2 = new ArrayList();
        this.fragments = arrayList2;
        PersonalMTopicFragment personalMTopicFragment = new PersonalMTopicFragment();
        this.mPersonalMTopicFragment = personalMTopicFragment;
        arrayList2.add(personalMTopicFragment);
        List<Fragment> list = this.fragments;
        PersonalMModeFragment personalMModeFragment = new PersonalMModeFragment();
        this.mPersonalMModeFragment = personalMModeFragment;
        list.add(personalMModeFragment);
        FragmentAdapter fragmentAdapter = new FragmentAdapter(getSupportFragmentManager(), this.fragments, arrayList);
        this.mFragmentAdapter = fragmentAdapter;
        this.mFragmentViewPager.setAdapter(fragmentAdapter);
        this.mTabs.setupWithViewPager(this.mFragmentViewPager);
        if (this.mTabs.getTabCount() > 0) {
            changeTabTextView(this.mTabs.getTabAt(0), true);
        }
        UserModel userModel = MyApplication.getApplication().getmUserModel();
        if (userModel != null) {
            initUserInfo(userModel);
        }
        this.pictureResultManager = new PictureResultManager(this);
        getMyInfo();
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initUserInfo(UserModel userModel) {
        Glide.with((FragmentActivity) this).load(userModel.getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into((RequestBuilder) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.7
            public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                PersonalDetailsActivity2.this.mImgHead.setImageDrawable(drawable);
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
            }
        });
        this.mImgHead.setKing(userModel.getKing(), this, userModel.getGradeIcon());
        this.mDetailsTxtName.setText(userModel.getName());
        this.mDetailsTxtConstellation.setText(userModel.getConstellation());
        this.mDetailsTxtId.setText(userModel.getUserId() + "");
        this.mDetailsTxtFans.setText(userModel.getFansCount() + "");
        this.mDetailsTxtFocus.setText(userModel.getFocusCount() + "");
        if (TextUtils.isEmpty(userModel.getBios())) {
            this.mDetailsTxtBios.setVisibility(8);
        } else {
            this.mDetailsTxtBios.setText(LanguageUtil.getLanguageByKey(this, "bio") + ": " + userModel.getBios());
            this.mDetailsTxtBios.setVisibility(0);
        }
        if (userModel.getGender() == 1) {
            this.mDetailsImgGender.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_details_men));
        } else {
            int gender = userModel.getGender();
            ImageView imageView = this.mDetailsImgGender;
            if (gender == 2) {
                imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_details_transgender));
            } else {
                imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_details_women));
            }
        }
        int interestedIn = userModel.getInterestedIn();
        LinearLayout linearLayout = this.mLayoutInterestedIn;
        if (interestedIn < 0) {
            linearLayout.setVisibility(8);
        } else {
            linearLayout.setVisibility(0);
            if (userModel.getInterestedIn() == 2) {
                LanguageUtil.setText(this, this.mTxtInterestedIn, "Trans");
            } else if (userModel.getInterestedIn() == 1) {
                LanguageUtil.setText(this, this.mTxtInterestedIn, "Guys");
            } else if (userModel.getInterestedIn() == 0) {
                LanguageUtil.setText(this, this.mTxtInterestedIn, "Girls");
            }
        }
        if (TextUtils.isEmpty(userModel.getVoice())) {
            this.isVoiceStauts = 0;
            this.mImgVoice.setImageResource(R.mipmap.ic_mic);
            this.mImgVoiceB.setVisibility(8);
            this.mTxtVoiceTime.setText(LanguageUtil.getLanguageByKey(this, "voiceRecording"));
        } else {
            this.isVoiceStauts = 1;
            this.mImgVoice.setImageResource(R.mipmap.ic_listen);
            this.mImgVoiceB.setVisibility(0);
            this.mTxtVoiceTime.setText(userModel.getVoiceLength() + "''");
        }
        boolean isEmpty = TextUtils.isEmpty(userModel.getCity());
        TextView textView = this.mDetailsTxtLocation;
        if (isEmpty) {
            textView.setText(userModel.getCountry());
        } else {
            textView.setText(userModel.getCountry() + ", " + userModel.getCity());
        }
        this.photos = new UserPhotosModel[0];
        this.imageList = new ArrayList();
        this.pagerView = new ArrayList();
        this.mLayoutDots.removeAllViews();
        if (userModel.getuPhotos() != null && userModel.getuPhotos().length > 0) {
            this.photos = userModel.getuPhotos();
            LayoutInflater from = LayoutInflater.from(this);
            for (int i = 0; i < this.photos.length; i++) {
                this.pagerView.add(from.inflate(R.layout.item_details_image, (ViewGroup) null, false));
                ImageView imageView2 = new ImageView(this);
                LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(MeasureUtil.dip2px(this, 7.0f), MeasureUtil.dip2px(this, 7.0f));
                layoutParams.setMarginEnd(MeasureUtil.dip2px(this, 6.0f));
                imageView2.setLayoutParams(layoutParams);
                if (i == 0) {
                    imageView2.setBackground(ContextCompat.getDrawable(this, R.drawable.dot_select));
                } else {
                    imageView2.setBackground(ContextCompat.getDrawable(this, R.drawable.dot_unselect));
                }
                this.mLayoutDots.addView(imageView2);
                this.imageList.add(this.photos[i].getUrl());
            }
        }
        this.mViewPager.setAdapter(new UserPhotosPagerAdapter(this, this.pagerView, 1, false, this.imageList, this.photos, userModel.getUserId()));
    }

    private void initView() {
        this.mLayoutRoot = (FrameLayout) findViewById(R.id.layout_root);
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtEdit = (TextView) findViewById(R.id.title_txt_edit);
        this.mLayoutPhoto = (RelativeLayout) findViewById(R.id.details_layout_photo);
        this.mViewPager = (ViewPager) findViewById(R.id.details_view_pager);
        this.mLayoutDots = (LinearLayout) findViewById(R.id.details_layout_dots);
        this.mLayoutVoice = (LinearLayout) findViewById(R.id.details_layout_voice);
        this.mImgVoice = (ImageView) findViewById(R.id.details_img_voice);
        this.mImgVoiceB = (ImageView) findViewById(R.id.details_img_voice_b);
        this.mTxtVoiceTime = (TextView) findViewById(R.id.details_txt_voice_time);
        this.mDetailsTxtName = (TextView) findViewById(R.id.details_txt_name);
        this.mDetailsTxtConstellation = (TextView) findViewById(R.id.details_txt_constellation);
        this.mDetailsTxtId = (TextView) findViewById(R.id.details_txt_id);
        this.mDetailsTxtLocation = (TextView) findViewById(R.id.details_txt_location);
        this.mDetailsTxtFans = (TextView) findViewById(R.id.details_txt_fans);
        this.mDetailsTxtFocus = (TextView) findViewById(R.id.details_txt_focus);
        this.mDetailsTxtIdLabel = (TextView) findViewById(R.id.details_txt_id_label);
        this.mDetailsTxtFansLabel = (TextView) findViewById(R.id.details_txt_fans_label);
        this.mDetailsTxtFocusLabel = (TextView) findViewById(R.id.details_txt_focus_label);
        this.mDetailsImgGender = (ImageView) findViewById(R.id.details_img_gender);
        this.mLayoutInterestedIn = (LinearLayout) findViewById(R.id.details_layout_interestedIn);
        this.mTxtInterestedInLabel = (TextView) findViewById(R.id.details_txt_interestedIn_label);
        this.mTxtInterestedIn = (TextView) findViewById(R.id.details_txt_interestedIn);
        this.mTabs = (TabLayout) findViewById(R.id.details_tabs);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.details_refreshLayout);
        this.mFragmentViewPager = (ViewPager) findViewById(R.id.details_fragment_view_pager);
        this.mAppbarLayout = (AppBarLayout) findViewById(R.id.appbar_layout);
        this.mToolbar = (Toolbar) findViewById(R.id.toolbar);
        this.mImgHead = (CircleImageView) findViewById(R.id.details_img_head);
        this.mLayoutEdit = (LinearLayout) findViewById(R.id.details_layout_edit);
        this.mLayoutChat = (LinearLayout) findViewById(R.id.details_layout_chat);
        this.mLayoutFollow = (LinearLayout) findViewById(R.id.details_layout_follow);
        this.mTxtEdit = (TextView) findViewById(R.id.details_txt_edit);
        this.mTxtChat = (TextView) findViewById(R.id.details_txt_chat);
        this.mTxtFollow = (TextView) findViewById(R.id.details_txt_follow);
        this.mTxtLocationLabel = (TextView) findViewById(R.id.details_txt_location_label);
        this.mDetailsTxtBios = (TextView) findViewById(R.id.details_txt_bios);
        this.mToolbarPlaceholder = (Toolbar) findViewById(R.id.title_bar);
        LanguageUtil.setText(this, this.mTitleTxtEdit, "editPersonal");
        this.mDetailsTxtIdLabel.setText(LanguageUtil.getLanguageByKey(this, NativeProtocol.BRIDGE_ARG_APP_NAME_STRING) + " ID:");
        LanguageUtil.setText(this, this.mDetailsTxtFansLabel, "fansCount");
        LanguageUtil.setText(this, this.mDetailsTxtFocusLabel, "following");
        LanguageUtil.setText(this, this.mTxtChat, "chat");
        LanguageUtil.setText(this, this.mTxtFollow, "followCount");
        this.mTxtLocationLabel.setText(LanguageUtil.getLanguageByKey(this, UserDataStore.COUNTRY) + ":");
        this.mTxtInterestedInLabel.setText(LanguageUtil.getLanguageByKey(this, "InterestedIn") + ":");
        this.mDetailsTxtBios.setText(LanguageUtil.getLanguageByKey(this, "bio") + ":");
        LanguageUtil.setText(this, this.mTxtEdit, "editPersonal");
        this.mTitleTxtEdit.setVisibility(8);
        this.mLayoutChat.setVisibility(8);
        this.mLayoutFollow.setVisibility(8);
        StatusBarUtil.fitTitleBar(this, this.mToolbar);
        StatusBarUtil.fitTitleBar(this, this.mToolbarPlaceholder);
        MeasureUtil.buildNavigationBarHeight(this.mLayoutRoot, false);
    }

    private void playVoice(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        releasePlay();
        try {
            MediaPlayer mediaPlayer = new MediaPlayer();
            this.mediaPlayer = mediaPlayer;
            mediaPlayer.setDataSource(str);
            this.mediaPlayer.setOnPreparedListener(this);
            this.mediaPlayer.setOnCompletionListener(this);
            this.mediaPlayer.setOnErrorListener(this);
            this.mediaPlayer.prepareAsync();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_USER_INFO_CHANGE);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void releasePlay() {
        MediaPlayer mediaPlayer = this.mediaPlayer;
        if (mediaPlayer != null) {
            mediaPlayer.release();
            this.mediaPlayer = null;
        }
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mLayoutEdit.setOnClickListener(this);
        this.mTitleTxtEdit.setOnClickListener(this);
        this.mImgHead.setOnClickListener(this);
        this.mLayoutVoice.setOnClickListener(this);
        this.mViewPager.addOnPageChangeListener(this);
        PersonalMTopicFragment personalMTopicFragment = this.mPersonalMTopicFragment;
        if (personalMTopicFragment != null) {
            personalMTopicFragment.setmListener(new PersonalMTopicFragment.OnItemCountChangeListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.1
                @Override // com.hismith.hismithapp.fragment.PersonalMTopicFragment.OnItemCountChangeListener
                public void getItemCount(String str) {
                    if (PersonalDetailsActivity2.this.mTabs.getTabCount() > 0) {
                        PersonalDetailsActivity2.this.mTabs.getTabAt(0).setText(LanguageUtil.getLanguageByKey(PersonalDetailsActivity2.this, "topicTitle") + " " + str);
                    }
                }
            });
        }
        PersonalMModeFragment personalMModeFragment = this.mPersonalMModeFragment;
        if (personalMModeFragment != null) {
            personalMModeFragment.setmListener(new PersonalMTopicFragment.OnItemCountChangeListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.2
                @Override // com.hismith.hismithapp.fragment.PersonalMTopicFragment.OnItemCountChangeListener
                public void getItemCount(String str) {
                    if (PersonalDetailsActivity2.this.mTabs.getTabCount() > 1) {
                        PersonalDetailsActivity2.this.mTabs.getTabAt(1).setText(LanguageUtil.getLanguageByKey(PersonalDetailsActivity2.this, "mode") + " " + str);
                    }
                }
            });
        }
        this.mTabs.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.3
            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabReselected(TabLayout.Tab tab) {
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabSelected(TabLayout.Tab tab) {
                PersonalDetailsActivity2.this.changeTabTextView(tab, true);
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabUnselected(TabLayout.Tab tab) {
                PersonalDetailsActivity2.this.changeTabTextView(tab, false);
            }
        });
        this.mAppbarLayout.addOnOffsetChangedListener(new AppBarLayout.OnOffsetChangedListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.4
            @Override // com.google.android.material.appbar.AppBarLayout.OnOffsetChangedListener, com.google.android.material.appbar.AppBarLayout.BaseOnOffsetChangedListener
            public void onOffsetChanged(AppBarLayout appBarLayout, int i) {
                int abs = (int) Math.abs((255.0f / appBarLayout.getTotalScrollRange()) * i);
                PersonalDetailsActivity2.this.mTitleImgBack.getBackground().mutate().setAlpha(255 - abs);
                PersonalDetailsActivity2.this.mToolbar.setBackgroundColor(Color.argb(abs, PersonalDetailsActivity2.this.getResources().getInteger(R.integer.toolbar_red), PersonalDetailsActivity2.this.getResources().getInteger(R.integer.toolbar_green), PersonalDetailsActivity2.this.getResources().getInteger(R.integer.toolbar_blue)));
            }
        });
        this.mRefreshLayout.setOnRefreshListener(new OnRefreshListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.5
            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PersonalDetailsActivity2.this.getMyInfo();
                if (PersonalDetailsActivity2.this.mPersonalMTopicFragment != null) {
                    PersonalDetailsActivity2.this.mPersonalMTopicFragment.onRefresh();
                }
                if (PersonalDetailsActivity2.this.mPersonalMModeFragment != null) {
                    PersonalDetailsActivity2.this.mPersonalMModeFragment.onRefresh();
                }
            }
        });
        this.mRefreshLayout.setOnMultiListener(new SimpleMultiListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.6
            @Override // com.scwang.smart.refresh.layout.simple.SimpleMultiListener, com.scwang.smart.refresh.layout.listener.OnMultiListener
            public void onHeaderMoving(RefreshHeader refreshHeader, boolean z, float f, int i, int i2, int i3) {
                LogUtil.e("onHeaderMoving isDragging:" + z + ", percent:" + f + ", offset:" + i + ", headerHeight:" + i2 + ", maxDragHeight:" + i3 + ", layout:" + PersonalDetailsActivity2.this.mLayoutPhoto.getHeight() + ", " + PersonalDetailsActivity2.this.mLayoutPhoto.getWidth());
                PersonalDetailsActivity2.this.mToolbar.setAlpha(1.0f - Math.min(f, 1.0f));
                ViewGroup.LayoutParams layoutParams = PersonalDetailsActivity2.this.mLayoutPhoto.getLayoutParams();
                layoutParams.height = MeasureUtil.dip2px(PersonalDetailsActivity2.this, 210.0f) + i;
                PersonalDetailsActivity2.this.mLayoutPhoto.setLayoutParams(layoutParams);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setMyHead(final String str, String str2) {
        this.baseRequest.getApi().setUserHead((String) SharedPreferencesUtil.get(this, Constans.Token), str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.11
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                int success = httpModel.getSuccess();
                PersonalDetailsActivity2 personalDetailsActivity2 = PersonalDetailsActivity2.this;
                if (success != 1) {
                    ToastUtil.showShortToastCenter(PersonalDetailsActivity2.this, LanguageUtil.getLanguageByKey(personalDetailsActivity2, "saveFailed"));
                    return;
                }
                ToastUtil.showShortToastCenter(PersonalDetailsActivity2.this, LanguageUtil.getLanguageByKey(personalDetailsActivity2, "saveed"));
                if (MyApplication.getApplication().getmUserModel() != null) {
                    MyApplication.getApplication().getmUserModel().setHeadIcon(str);
                }
                PersonalDetailsActivity2.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
            }
        });
    }

    private void showHeadAction() {
        ChoosePhotoDialog choosePhotoDialog = new ChoosePhotoDialog(this) { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.9
            @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
            public void btnClickDelete() {
            }

            @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
            public void btnClickPhoto() {
                PersonalDetailsActivity2.this.pictureResultManager.openGallery(new PictureResultManager.OnGalleryListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.9.2
                    @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
                    public void onGalleryResult(Uri uri) {
                        if (uri != null) {
                            PersonalDetailsActivity2.this.uploadFile(uri);
                            Glide.with((FragmentActivity) PersonalDetailsActivity2.this).load(uri).placeholder(R.mipmap.ic_personal_head).into(PersonalDetailsActivity2.this.mImgHead);
                        }
                    }

                    @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
                    public void onGalleryResultFile(String str) {
                        if (TextUtils.isEmpty(str)) {
                            return;
                        }
                        PersonalDetailsActivity2.this.uploadFile(new File(str));
                        RequestManager with = Glide.with((FragmentActivity) PersonalDetailsActivity2.this);
                        boolean isContent = PictureMimeType.isContent(str);
                        Object obj = str;
                        if (isContent) {
                            obj = Uri.parse(str);
                        }
                        with.load(obj).placeholder(R.mipmap.ic_personal_head).into(PersonalDetailsActivity2.this.mImgHead);
                    }
                });
            }

            @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
            public void btnClickShoot() {
                PictureSelectorManager.getInstance(PersonalDetailsActivity2.this).openCamera(new PictureSelectorManager.OnCameraListener() { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.9.1
                    @Override // com.hismith.hismithapp.picker.PictureSelectorManager.OnCameraListener
                    public void onCameraResult(List<LocalMedia> list) {
                        for (int i = 0; i < list.size(); i++) {
                            if (list.get(i).isCompressed()) {
                                String compressPath = list.get(i).getCompressPath();
                                LogUtil.e("新图片库拍摄压缩后：" + compressPath + ", 文件是否存在：" + Uri.parse(compressPath).getPath());
                                PersonalDetailsActivity2.this.uploadFile(new File(compressPath));
                                RequestManager with = Glide.with((FragmentActivity) PersonalDetailsActivity2.this);
                                boolean isContent = PictureMimeType.isContent(compressPath);
                                Object obj = compressPath;
                                if (isContent) {
                                    obj = Uri.parse(compressPath);
                                }
                                with.load(obj).placeholder(R.mipmap.ic_personal_head).into(PersonalDetailsActivity2.this.mImgHead);
                            } else {
                                String realPath = list.get(i).getRealPath();
                                PersonalDetailsActivity2.this.uploadFile(new File(realPath));
                                RequestManager with2 = Glide.with((FragmentActivity) PersonalDetailsActivity2.this);
                                boolean isContent2 = PictureMimeType.isContent(realPath);
                                Object obj2 = realPath;
                                if (isContent2) {
                                    obj2 = Uri.parse(realPath);
                                }
                                with2.load(obj2).placeholder(R.mipmap.ic_personal_head).into(PersonalDetailsActivity2.this.mImgHead);
                            }
                        }
                    }
                });
            }
        };
        choosePhotoDialog.setHideDelete(true);
        choosePhotoDialog.show();
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void upload(Map<String, RequestBody> map, final String str) {
        this.fileRequest.getApi().uploadFile(map).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UploadModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.PersonalDetailsActivity2.12
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UploadModel uploadModel) {
                if (uploadModel.getSuccess() == 1) {
                    String fileUrl = uploadModel.getFileUrl();
                    LogUtil.e("file:" + str + ", fileUrl:" + fileUrl);
                    PersonalDetailsActivity2.this.setMyHead(str, fileUrl);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void uploadFile(Uri uri) {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        if (uri != null) {
            LogUtil.e("上传图片file：" + uri);
            String fileNameFromUri = RequestBodyUtil.getFileNameFromUri(this, uri);
            HashMap hashMap = new HashMap();
            RequestBodyUtil.addMultiPart(this, hashMap, Constans.Token, str);
            RequestBodyUtil.addMultiPart(this, hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
            RequestBodyUtil.addMultiPart(this, hashMap, "requestId", fileNameFromUri);
            RequestBodyUtil.addMultiPart(this, hashMap, "fileType", "image");
            RequestBodyUtil.addMultiPartUri(this, hashMap, fileNameFromUri, uri);
            upload(hashMap, uri.toString());
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void uploadFile(File file) {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        if (file == null || !file.exists()) {
            return;
        }
        LogUtil.e("上传图片file：" + file.getPath());
        HashMap hashMap = new HashMap();
        RequestBodyUtil.addMultiPart(this, hashMap, Constans.Token, str);
        RequestBodyUtil.addMultiPart(this, hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
        RequestBodyUtil.addMultiPart(this, hashMap, "requestId", file.getName());
        RequestBodyUtil.addMultiPart(this, hashMap, "fileType", "image");
        RequestBodyUtil.addMultiPart(this, hashMap, "file", file);
        upload(hashMap, file.getPath());
    }

    public void changeTabTextView(TabLayout.Tab tab, boolean z) {
        if (tab.getCustomView() == null) {
            tab.setCustomView(R.layout.tab_layout_text);
        }
        TextView textView = (TextView) tab.getCustomView().findViewById(android.R.id.text1);
        if (z) {
            textView.setTextAppearance(this, R.style.TabLayoutBoldTextSize);
        } else {
            textView.setTextAppearance(this, R.style.TabLayoutNormalTextSize);
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.title_txt_edit || id == R.id.details_layout_edit) {
            startActivity(new Intent(this, (Class<?>) DetailsEditActivity.class));
            return;
        }
        if (id == R.id.details_img_head) {
            showHeadAction();
            return;
        }
        if (id == R.id.details_layout_voice) {
            int i = this.isVoiceStauts;
            if (i == 1) {
                playVoice(MyApplication.getApplication().getmUserModel().getVoice());
                this.isVoiceStauts = 2;
                Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.ic_voice_play_b)).placeholder(R.mipmap.ic_wave).into(this.mImgVoiceB);
            } else if (i == 2) {
                releasePlay();
                this.isVoiceStauts = 1;
                this.mImgVoiceB.setImageResource(R.mipmap.ic_wave);
            } else if (i == 0) {
                startActivity(new Intent(this, (Class<?>) VoiceSignatureActivity.class));
            }
        }
    }

    @Override // android.media.MediaPlayer.OnCompletionListener
    public void onCompletion(MediaPlayer mediaPlayer) {
        this.isVoiceStauts = 1;
        this.mImgVoiceB.setImageResource(R.mipmap.ic_wave);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_personal_details2);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        releasePlay();
        unRegisterReceiver();
    }

    @Override // android.media.MediaPlayer.OnErrorListener
    public boolean onError(MediaPlayer mediaPlayer, int i, int i2) {
        LogUtil.e("OnError what:" + i + ", extra:" + i2);
        return false;
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageScrollStateChanged(int i) {
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageScrolled(int i, float f, int i2) {
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageSelected(int i) {
        LogUtil.e("pageSelected position:" + i);
        for (int i2 = 0; i2 < this.mLayoutDots.getChildCount(); i2++) {
            ImageView imageView = (ImageView) this.mLayoutDots.getChildAt(i2);
            if (i == i2) {
                imageView.setBackground(ContextCompat.getDrawable(this, R.drawable.dot_select));
            } else {
                imageView.setBackground(ContextCompat.getDrawable(this, R.drawable.dot_unselect));
            }
        }
    }

    @Override // android.media.MediaPlayer.OnPreparedListener
    public void onPrepared(MediaPlayer mediaPlayer) {
        try {
            mediaPlayer.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
