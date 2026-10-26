package com.hismith.hismithapp;

import android.content.Intent;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager.widget.PagerAdapter;
import androidx.viewpager.widget.ViewPager;
import com.bumptech.glide.Glide;
import com.facebook.appevents.UserDataStore;
import com.facebook.internal.NativeProtocol;
import com.google.android.material.appbar.AppBarLayout;
import com.google.android.material.tabs.TabLayout;
import com.google.gson.Gson;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.adapter.UserPhotosPagerAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.db.DBMessageUtil;
import com.hismith.hismithapp.db.DBUserUtil;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.dialog.ReportSelectDialog;
import com.hismith.hismithapp.dialog.SelectFollowDialog;
import com.hismith.hismithapp.fragment.PersonalMModeFragment;
import com.hismith.hismithapp.fragment.PersonalMTopicFragment;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.model.BlockModel;
import com.hismith.hismithapp.model.FollowModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.model.UserPhotosModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.GlideImageUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StatusBarUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshHeader;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshListener;
import com.scwang.smart.refresh.layout.simple.SimpleMultiListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class UserDetailsActivity2 extends BaseActivity implements View.OnClickListener, ViewPager.OnPageChangeListener, MediaPlayer.OnPreparedListener, MediaPlayer.OnCompletionListener {
    public static int REQUEST_USER_INFO = 17;
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbWrite;
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
    private ImageView mImgFollow;
    private CircleImageView mImgHead;
    private ImageView mImgVoice;
    private ImageView mImgVoiceB;
    private LinearLayout mLayoutChat;
    private LinearLayout mLayoutDots;
    private LinearLayout mLayoutEdit;
    private LinearLayout mLayoutFollow;
    private LinearLayout mLayoutInterestedIn;
    private RelativeLayout mLayoutPhoto;
    private LinearLayout mLayoutVoice;
    private PersonalMModeFragment mPersonalMModeFragment;
    private PersonalMTopicFragment mPersonalMTopicFragment;
    private ProgressBar mProgressFollow;
    private SmartRefreshLayout mRefreshLayout;
    private TabLayout mTabs;
    private ImageView mTitleImgBack;
    private ImageView mTitleImgMore;
    private TextView mTitleTxtName;
    private Toolbar mToolbar;
    private Toolbar mToolbarPlaceholder;
    private TextView mTxtChat;
    private TextView mTxtFollow;
    private TextView mTxtInterestedIn;
    private TextView mTxtInterestedInLabel;
    private TextView mTxtLocationLabel;
    private TextView mTxtVoiceTime;
    private UserModel mUserModel;
    private ViewPager mViewPager;
    private MediaPlayer mediaPlayer;
    private List<View> pagerView;
    private UserPhotosModel[] photos;
    private boolean isHasSexPhotos = false;
    private int userId = 0;
    private int isVoiceStauts = 0;
    PagerAdapter mPagerAdapter = new PagerAdapter() { // from class: com.hismith.hismithapp.UserDetailsActivity2.7
        @Override // androidx.viewpager.widget.PagerAdapter
        public void destroyItem(View view, int i, Object obj) {
            if (i < UserDetailsActivity2.this.pagerView.size()) {
                ((ViewPager) view).removeView((View) UserDetailsActivity2.this.pagerView.get(i));
            }
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public int getCount() {
            return UserDetailsActivity2.this.pagerView.size();
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public Object instantiateItem(View view, final int i) {
            ImageView imageView = (ImageView) ((View) UserDetailsActivity2.this.pagerView.get(i)).findViewById(R.id.item_details_image);
            imageView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.UserDetailsActivity2.7.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view2) {
                    int viewSex = MyApplication.getApplication().getmUserModel() != null ? MyApplication.getApplication().getmUserModel().getViewSex() : 0;
                    if (UserDetailsActivity2.this.isHasSexPhotos && viewSex == 0) {
                        UserDetailsActivity2.this.startActivity(new Intent(UserDetailsActivity2.this, (Class<?>) SexSettingsActivity.class));
                    } else {
                        if (UserDetailsActivity2.this.imageList == null || UserDetailsActivity2.this.imageList.size() <= i) {
                            return;
                        }
                        ImagePreview.getInstance().setContext(UserDetailsActivity2.this).setUserId(UserDetailsActivity2.this.mUserModel.getUserId()).setIndex(i).setImageList(UserDetailsActivity2.this.imageList).setFolderName("hismith/image").setEnableDragClose(true).start();
                    }
                }
            });
            int viewSex = MyApplication.getApplication().getmUserModel() != null ? MyApplication.getApplication().getmUserModel().getViewSex() : 0;
            UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
            GlideImageUtil.loadBlurredImage(userDetailsActivity2, userDetailsActivity2.photos[i].getUrl(), imageView, 0, 0, UserDetailsActivity2.this.photos[i].getSex(), viewSex);
            ((ViewPager) view).addView((View) UserDetailsActivity2.this.pagerView.get(i));
            return UserDetailsActivity2.this.pagerView.get(i);
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public boolean isViewFromObject(View view, Object obj) {
            return view == obj;
        }
    };

    private void closeDB() {
        SQLiteDatabase sQLiteDatabase = this.dbWrite;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
        }
        DatabaseHelper databaseHelper = this.dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(String str, String str2) {
        if (this.mUserModel == null) {
            return;
        }
        this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.mUserModel.getUserId()), "user", str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.UserDetailsActivity2.11
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(UserDetailsActivity2.this, LanguageUtil.getLanguageByKey(UserDetailsActivity2.this, "reportedSuccess"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void follow(int i) {
        if (this.mUserModel == null) {
            return;
        }
        boolean z = false;
        this.mLayoutFollow.setEnabled(false);
        ImageView imageView = this.mImgFollow;
        if (imageView != null && imageView.getVisibility() == 0) {
            this.mImgFollow.setVisibility(4);
        }
        this.mTxtFollow.setVisibility(4);
        this.mProgressFollow.setVisibility(0);
        this.mLayoutFollow.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_corners_border_style_15));
        this.baseRequest.getApi().follow((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.mUserModel.getUserId()), Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FollowModel>(this, this.progressDialog, z) { // from class: com.hismith.hismithapp.UserDetailsActivity2.9
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
                userDetailsActivity2.initFollow(userDetailsActivity2.mUserModel.getIsFans());
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
                userDetailsActivity2.initFollow(userDetailsActivity2.mUserModel.getIsFans());
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FollowModel followModel) {
                if (followModel.getSuccess() == 1) {
                    UserDetailsActivity2.this.mUserModel.setIsFans(followModel.getFollow().intValue());
                    if (followModel.getFollow().intValue() == 1) {
                        if (MyApplication.getApplication().getmUserModel() != null) {
                            MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() + 1);
                            UserDetailsActivity2.this.mUserModel.setFansCount(UserDetailsActivity2.this.mUserModel.getFansCount() + 1);
                            UserDetailsActivity2.this.mDetailsTxtFans.setText(UserDetailsActivity2.this.mUserModel.getFansCount() + "");
                            UserDetailsActivity2.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                            return;
                        }
                        return;
                    }
                    if (MyApplication.getApplication().getmUserModel() != null) {
                        MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() - 1);
                        UserDetailsActivity2.this.mUserModel.setFansCount(UserDetailsActivity2.this.mUserModel.getFansCount() - 1);
                        UserDetailsActivity2.this.mDetailsTxtFans.setText(UserDetailsActivity2.this.mUserModel.getFansCount() + "");
                        UserDetailsActivity2.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getUserInfo() {
        if (this.userId == 0) {
            return;
        }
        this.baseRequest.getApi().getUserInfo((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.userId)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.UserDetailsActivity2.8
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                if (UserDetailsActivity2.this.mRefreshLayout != null) {
                    UserDetailsActivity2.this.mRefreshLayout.finishRefresh();
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                if (UserDetailsActivity2.this.mRefreshLayout != null) {
                    UserDetailsActivity2.this.mRefreshLayout.finishRefresh();
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel.getSuccess() == 1) {
                    UserDetailsActivity2.this.mUserModel = userModel;
                    UserDetailsActivity2.this.initUserInfo(userModel);
                    UserDetailsActivity2.this.initFollow(userModel.getIsFans());
                    DBUserUtil.insertDBMessageUserByUserDetails(userModel);
                }
            }
        });
    }

    private void init() {
        if (getIntent() != null) {
            this.userId = getIntent().getIntExtra("userId", 0);
        }
        this.baseRequest = new RetrofitRequest();
        DatabaseHelper databaseHelper = new DatabaseHelper(this);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
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
        if (this.userId != 0) {
            PersonalMTopicFragment personalMTopicFragment = new PersonalMTopicFragment(this.userId + "");
            this.mPersonalMTopicFragment = personalMTopicFragment;
            arrayList2.add(personalMTopicFragment);
            List<Fragment> list = this.fragments;
            PersonalMModeFragment personalMModeFragment = new PersonalMModeFragment(this.userId + "");
            this.mPersonalMModeFragment = personalMModeFragment;
            list.add(personalMModeFragment);
            FragmentAdapter fragmentAdapter = new FragmentAdapter(getSupportFragmentManager(), this.fragments, arrayList);
            this.mFragmentAdapter = fragmentAdapter;
            this.mFragmentViewPager.setAdapter(fragmentAdapter);
            this.mTabs.setupWithViewPager(this.mFragmentViewPager);
            if (this.mTabs.getTabCount() > 0) {
                changeTabTextView(this.mTabs.getTabAt(0), true);
            }
        }
        getUserInfo();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initFollow(int i) {
        ImageView imageView = this.mImgFollow;
        if (i == 1) {
            if (imageView != null) {
                imageView.setVisibility(8);
            }
            this.mTxtFollow.setVisibility(0);
            LanguageUtil.setText(this, this.mTxtFollow, "followed");
            this.mTxtFollow.setTextColor(ContextCompat.getColor(this, R.color.app_style_color));
            this.mLayoutFollow.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_corners_border_style_15));
        } else {
            if (imageView != null) {
                imageView.setVisibility(0);
            }
            this.mTxtFollow.setVisibility(0);
            LanguageUtil.setText(this, this.mTxtFollow, "followCount");
            this.mTxtFollow.setTextColor(ContextCompat.getColor(this, R.color.white));
            this.mLayoutFollow.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_edit_style_15));
        }
        this.mLayoutFollow.setEnabled(true);
        this.mProgressFollow.setVisibility(8);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initUserInfo(UserModel userModel) {
        int viewSex = MyApplication.getApplication().getmUserModel() != null ? MyApplication.getApplication().getmUserModel().getViewSex() : 0;
        GlideImageUtil.loadBlurredImage(this, userModel.getHeadIcon(), this.mImgHead, R.mipmap.ic_personal_head, 0, userModel.getHeadSex(), viewSex);
        this.mImgHead.setKing(userModel.getKing(), this, userModel.getGradeIcon());
        this.mTitleTxtName.setText(userModel.getName());
        boolean isEmpty = TextUtils.isEmpty(userModel.getAlias());
        TextView textView = this.mDetailsTxtName;
        if (isEmpty) {
            textView.setText(userModel.getName());
        } else {
            textView.setText(userModel.getAlias());
        }
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
            this.mLayoutVoice.setVisibility(8);
        } else {
            this.isVoiceStauts = 1;
            this.mImgVoice.setImageResource(R.mipmap.ic_listen);
            this.mImgVoiceB.setVisibility(0);
            this.mTxtVoiceTime.setText(userModel.getVoiceLength() + "''");
            this.mLayoutVoice.setVisibility(0);
        }
        boolean isEmpty2 = TextUtils.isEmpty(userModel.getCity());
        TextView textView2 = this.mDetailsTxtLocation;
        if (isEmpty2) {
            textView2.setText(userModel.getCountry());
        } else {
            textView2.setText(userModel.getCountry() + ", " + userModel.getCity());
        }
        this.photos = new UserPhotosModel[0];
        this.isHasSexPhotos = false;
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
                if (this.photos[i].getSex() == 1) {
                    this.isHasSexPhotos = true;
                }
                this.imageList.add(this.photos[i].getUrl());
            }
        }
        this.mViewPager.setAdapter(new UserPhotosPagerAdapter(this, this.pagerView, viewSex, this.isHasSexPhotos, this.imageList, this.photos, userModel.getUserId()));
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleImgMore = (ImageView) findViewById(R.id.title_img_more);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
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
        this.mDetailsTxtBios = (TextView) findViewById(R.id.details_txt_bios);
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
        this.mTxtChat = (TextView) findViewById(R.id.details_txt_chat);
        this.mTxtFollow = (TextView) findViewById(R.id.details_txt_follow);
        this.mTxtLocationLabel = (TextView) findViewById(R.id.details_txt_location_label);
        this.mProgressFollow = (ProgressBar) findViewById(R.id.details_progress_follow);
        this.mToolbarPlaceholder = (Toolbar) findViewById(R.id.title_bar);
        this.mDetailsTxtIdLabel.setText(LanguageUtil.getLanguageByKey(this, NativeProtocol.BRIDGE_ARG_APP_NAME_STRING) + " ID:");
        LanguageUtil.setText(this, this.mDetailsTxtFansLabel, "fansCount");
        LanguageUtil.setText(this, this.mDetailsTxtFocusLabel, "following");
        LanguageUtil.setText(this, this.mTxtChat, "chat");
        LanguageUtil.setText(this, this.mTxtFollow, "followCount");
        this.mTxtLocationLabel.setText(LanguageUtil.getLanguageByKey(this, UserDataStore.COUNTRY) + ":");
        this.mTxtInterestedInLabel.setText(LanguageUtil.getLanguageByKey(this, "InterestedIn") + ":");
        this.mLayoutEdit.setVisibility(8);
        this.mDetailsTxtBios.setText(LanguageUtil.getLanguageByKey(this, "bio") + ":");
        StatusBarUtil.fitTitleBar(this, this.mToolbar);
        StatusBarUtil.fitTitleBar(this, this.mToolbarPlaceholder);
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
            this.mediaPlayer.prepareAsync();
        } catch (Exception e) {
            e.printStackTrace();
        }
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
        this.mTitleImgMore.setOnClickListener(this);
        this.mImgHead.setOnClickListener(this);
        this.mLayoutVoice.setOnClickListener(this);
        this.mLayoutFollow.setOnClickListener(this);
        this.mLayoutChat.setOnClickListener(this);
        this.mViewPager.addOnPageChangeListener(this);
        PersonalMTopicFragment personalMTopicFragment = this.mPersonalMTopicFragment;
        if (personalMTopicFragment != null) {
            personalMTopicFragment.setmListener(new PersonalMTopicFragment.OnItemCountChangeListener() { // from class: com.hismith.hismithapp.UserDetailsActivity2.1
                @Override // com.hismith.hismithapp.fragment.PersonalMTopicFragment.OnItemCountChangeListener
                public void getItemCount(String str) {
                    if (UserDetailsActivity2.this.mTabs.getTabCount() > 0) {
                        UserDetailsActivity2.this.mTabs.getTabAt(0).setText(LanguageUtil.getLanguageByKey(UserDetailsActivity2.this, "topicTitle") + " " + str);
                    }
                }
            });
        }
        PersonalMModeFragment personalMModeFragment = this.mPersonalMModeFragment;
        if (personalMModeFragment != null) {
            personalMModeFragment.setmListener(new PersonalMTopicFragment.OnItemCountChangeListener() { // from class: com.hismith.hismithapp.UserDetailsActivity2.2
                @Override // com.hismith.hismithapp.fragment.PersonalMTopicFragment.OnItemCountChangeListener
                public void getItemCount(String str) {
                    if (UserDetailsActivity2.this.mTabs.getTabCount() > 1) {
                        UserDetailsActivity2.this.mTabs.getTabAt(1).setText(LanguageUtil.getLanguageByKey(UserDetailsActivity2.this, "mode") + " " + str);
                    }
                }
            });
        }
        this.mTabs.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() { // from class: com.hismith.hismithapp.UserDetailsActivity2.3
            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabReselected(TabLayout.Tab tab) {
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabSelected(TabLayout.Tab tab) {
                UserDetailsActivity2.this.changeTabTextView(tab, true);
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabUnselected(TabLayout.Tab tab) {
                UserDetailsActivity2.this.changeTabTextView(tab, false);
            }
        });
        this.mAppbarLayout.addOnOffsetChangedListener(new AppBarLayout.OnOffsetChangedListener() { // from class: com.hismith.hismithapp.UserDetailsActivity2.4
            @Override // com.google.android.material.appbar.AppBarLayout.OnOffsetChangedListener, com.google.android.material.appbar.AppBarLayout.BaseOnOffsetChangedListener
            public void onOffsetChanged(AppBarLayout appBarLayout, int i) {
                int abs = (int) Math.abs((255.0f / appBarLayout.getTotalScrollRange()) * i);
                int i2 = 255 - abs;
                UserDetailsActivity2.this.mTitleImgBack.getBackground().mutate().setAlpha(i2);
                UserDetailsActivity2.this.mTitleImgMore.getBackground().mutate().setAlpha(i2);
                UserDetailsActivity2.this.mTitleTxtName.setTextColor(Color.argb(abs, 255, 255, 255));
                UserDetailsActivity2.this.mToolbar.setBackgroundColor(Color.argb(abs, UserDetailsActivity2.this.getResources().getInteger(R.integer.toolbar_red), UserDetailsActivity2.this.getResources().getInteger(R.integer.toolbar_green), UserDetailsActivity2.this.getResources().getInteger(R.integer.toolbar_blue)));
            }
        });
        this.mRefreshLayout.setOnRefreshListener(new OnRefreshListener() { // from class: com.hismith.hismithapp.UserDetailsActivity2.5
            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                int i = UserDetailsActivity2.this.userId;
                UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
                if (i == 0) {
                    userDetailsActivity2.mRefreshLayout.finishRefresh(300);
                    return;
                }
                userDetailsActivity2.getUserInfo();
                if (UserDetailsActivity2.this.mPersonalMTopicFragment != null) {
                    UserDetailsActivity2.this.mPersonalMTopicFragment.onRefresh();
                }
                if (UserDetailsActivity2.this.mPersonalMModeFragment != null) {
                    UserDetailsActivity2.this.mPersonalMModeFragment.onRefresh();
                }
            }
        });
        this.mRefreshLayout.setOnMultiListener(new SimpleMultiListener() { // from class: com.hismith.hismithapp.UserDetailsActivity2.6
            @Override // com.scwang.smart.refresh.layout.simple.SimpleMultiListener, com.scwang.smart.refresh.layout.listener.OnMultiListener
            public void onHeaderMoving(RefreshHeader refreshHeader, boolean z, float f, int i, int i2, int i3) {
                UserDetailsActivity2.this.mToolbar.setAlpha(1.0f - Math.min(f, 1.0f));
                ViewGroup.LayoutParams layoutParams = UserDetailsActivity2.this.mLayoutPhoto.getLayoutParams();
                layoutParams.height = MeasureUtil.dip2px(UserDetailsActivity2.this, 210.0f) + i;
                UserDetailsActivity2.this.mLayoutPhoto.setLayoutParams(layoutParams);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setUserAlias(final String str) {
        if (this.mUserModel == null) {
            return;
        }
        this.baseRequest.getApi().setUserAlias((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.mUserModel.getUserId()), str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.UserDetailsActivity2.12
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    UserDetailsActivity2.this.mUserModel.setAlias(str);
                    DBUserUtil.updateDBUserByUserAlias(UserDetailsActivity2.this.mUserModel.getUserId(), str);
                    String str2 = str;
                    if (TextUtils.isEmpty(str2)) {
                        str2 = UserDetailsActivity2.this.mUserModel.getName();
                    }
                    DBUserUtil.updateDBReferenceUserName(UserDetailsActivity2.this.mUserModel.getUserId(), str2);
                    UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
                    userDetailsActivity2.initUserInfo(userDetailsActivity2.mUserModel);
                    Intent intent = new Intent();
                    intent.putExtra("alias", str2);
                    UserDetailsActivity2.this.setResult(1, intent);
                    ToastUtil.showShortToastCenter(UserDetailsActivity2.this, LanguageUtil.getLanguageByKey(UserDetailsActivity2.this, "saveed"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setUserBlackList(int i) {
        this.baseRequest.getApi().setUserBlackList((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.mUserModel.getUserId()), i).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<BlockModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.UserDetailsActivity2.10
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(BlockModel blockModel) {
                if (blockModel.getSuccess() == 1) {
                    UserDetailsActivity2.this.mUserModel.setIsBlocked(blockModel.getIsBlocked());
                    int isBlocked = blockModel.getIsBlocked();
                    UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
                    if (isBlocked != 1) {
                        ToastUtil.showShortToastCenter(UserDetailsActivity2.this, LanguageUtil.getLanguageByKey(userDetailsActivity2, "removed"));
                        return;
                    }
                    ToastUtil.showShortToastCenter(UserDetailsActivity2.this, LanguageUtil.getLanguageByKey(userDetailsActivity2, "successfullyBlocked"));
                    Intent intent = new Intent();
                    intent.putExtra(Constans.BlockUserId, UserDetailsActivity2.this.mUserModel.getUserId());
                    UserDetailsActivity2.this.setResult(108, intent);
                    UserDetailsActivity2.this.appManager.finishActivity();
                }
            }
        });
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

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        int intExtra;
        UserModel userModel;
        super.onActivityResult(i, i2, intent);
        try {
            if (i != REQUEST_USER_INFO || intent == null || (intExtra = intent.getIntExtra("IsBlocked", -1)) == -1 || (userModel = this.mUserModel) == null) {
                return;
            }
            userModel.setIsBlocked(intExtra);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.title_img_more) {
            if (this.mUserModel != null) {
                SelectFollowDialog selectFollowDialog = new SelectFollowDialog(this) { // from class: com.hismith.hismithapp.UserDetailsActivity2.13
                    @Override // com.hismith.hismithapp.dialog.SelectFollowDialog
                    public void btnClickBlack() {
                        int isBlocked = UserDetailsActivity2.this.mUserModel.getIsBlocked();
                        UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
                        if (isBlocked == 1) {
                            userDetailsActivity2.setUserBlackList(1);
                        } else {
                            userDetailsActivity2.setUserBlackList(0);
                        }
                    }

                    @Override // com.hismith.hismithapp.dialog.SelectFollowDialog
                    public void btnClickFollow() {
                        int isFans = UserDetailsActivity2.this.mUserModel.getIsFans();
                        UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
                        if (isFans == 1) {
                            userDetailsActivity2.follow(0);
                        } else {
                            userDetailsActivity2.follow(1);
                        }
                    }

                    @Override // com.hismith.hismithapp.dialog.SelectFollowDialog
                    public void btnClickNickName() {
                        UserDetailsActivity2 userDetailsActivity2 = UserDetailsActivity2.this;
                        DialogUtil.showEditDialog2(userDetailsActivity2, "remarks", userDetailsActivity2.mUserModel.getAlias(), null, null, new DialogUtil.OnEditContentCallback() { // from class: com.hismith.hismithapp.UserDetailsActivity2.13.1
                            @Override // com.hismith.hismithapp.utils.DialogUtil.OnEditContentCallback
                            public void onEditContent(String str) {
                                UserDetailsActivity2.this.setUserAlias(str);
                            }
                        });
                    }

                    @Override // com.hismith.hismithapp.dialog.SelectFollowDialog
                    public void btnClickReport() {
                        new ReportSelectDialog(UserDetailsActivity2.this) { // from class: com.hismith.hismithapp.UserDetailsActivity2.13.2
                            @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                            public void btnSubmit(String str, String str2) {
                                UserDetailsActivity2.this.complaint(str, str2);
                            }
                        }.showDialog();
                    }
                };
                selectFollowDialog.setFollow(this.mUserModel.getIsFans());
                selectFollowDialog.setIsBlocked(this.mUserModel.getIsBlocked());
                selectFollowDialog.show();
                return;
            }
            return;
        }
        if (id == R.id.details_img_head) {
            int viewSex = MyApplication.getApplication().getmUserModel() != null ? MyApplication.getApplication().getmUserModel().getViewSex() : 0;
            UserModel userModel = this.mUserModel;
            if (userModel != null && userModel.getHeadSex() == 1 && viewSex == 0) {
                startActivity(new Intent(this, (Class<?>) SexSettingsActivity.class));
                return;
            }
            return;
        }
        if (id == R.id.details_layout_follow) {
            UserModel userModel2 = this.mUserModel;
            if (userModel2 != null) {
                if (userModel2.getIsFans() == 1) {
                    follow(0);
                    return;
                } else {
                    follow(1);
                    return;
                }
            }
            return;
        }
        if (id == R.id.details_layout_chat) {
            UserModel userModel3 = this.mUserModel;
            if (userModel3 != null) {
                UserModel m614clone = userModel3.m614clone();
                if (!TextUtils.isEmpty(this.mUserModel.getAlias())) {
                    m614clone.setName(this.mUserModel.getAlias());
                }
                Intent intent = new Intent(this, (Class<?>) ChatActivity.class);
                intent.putExtra(Constans.UserModel, m614clone);
                intent.putExtra("firstChat", DBMessageUtil.getMessageCountByUser(m614clone.getUserId()) <= 0 ? 1 : 0);
                startActivityForResult(intent, REQUEST_USER_INFO);
                return;
            }
            return;
        }
        if (id == R.id.details_layout_voice) {
            int i = this.isVoiceStauts;
            if (i == 1) {
                playVoice(this.mUserModel.getVoice());
                this.isVoiceStauts = 2;
                Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.ic_voice_play_b)).placeholder(R.mipmap.ic_wave).into(this.mImgVoiceB);
            } else if (i == 2) {
                releasePlay();
                this.isVoiceStauts = 1;
                this.mImgVoiceB.setImageResource(R.mipmap.ic_wave);
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
        setContentView(R.layout.activity_user_details2);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        closeDB();
        releasePlay();
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
