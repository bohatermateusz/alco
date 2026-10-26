package com.hismith.hismithapp.fragment;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Message;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager.widget.ViewPager;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.google.gson.Gson;
import com.hismith.hismithapp.BlackListActivity;
import com.hismith.hismithapp.FansFocusActivity;
import com.hismith.hismithapp.FriendActivity;
import com.hismith.hismithapp.InfoActivity;
import com.hismith.hismithapp.MainActivity;
import com.hismith.hismithapp.PersonalActionActivity;
import com.hismith.hismithapp.PersonalCollectionActivity;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.PersonalMessageActivity;
import com.hismith.hismithapp.PlayStatistics2Activity;
import com.hismith.hismithapp.QRCodeActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.SettingActivity;
import com.hismith.hismithapp.adapter.ActionAdapter;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.model.ActionModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.PermissionUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.luck.picture.lib.permissions.PermissionConfig;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalFragment extends Fragment implements View.OnClickListener, ViewPager.OnPageChangeListener, ActionAdapter.OnItemClickListener {
    public static final int requestInfo = 112;
    private RetrofitRequest baseRequest;
    private Gson gson;
    private ActionAdapter mAdapter;
    private ImageView mImgArrow;
    private ImageView mImgGender;
    private CircleImageView mImgHead;
    private ImageView mImgScan;
    private ImageView mImgSetting;
    private LinearLayout mLayoutFans;
    private LinearLayout mLayoutFollow;
    private LinearLayout mLayoutFriend;
    private LinearLayout mLayoutHead;
    private LinearLayout mLayoutInfo;
    private GridLayoutManager mLayoutManager;
    private List<ActionModel> mList;
    private RecyclerView mRecyclerView;
    private TextView mTxtConstellation;
    private TextView mTxtFans;
    private TextView mTxtFansCount;
    private TextView mTxtFollow;
    private TextView mTxtFollowCount;
    private TextView mTxtFriend;
    private TextView mTxtFriendCount;
    private TextView mTxtName;
    private MainActivity mainActivity;
    private PermissionUtil permissionUtil;
    private boolean isCreateView = false;
    private boolean isFirst = true;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PersonalFragment.3
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (!action.equals(Constans.RECEIVER_MESSAGE_NEW)) {
                if (action.equals(Constans.RECEIVER_MESSAGE_READ)) {
                    PersonalFragment.this.initMessageRead(intent.getBooleanExtra("isUnRead", false));
                    return;
                }
                return;
            }
            intent.getIntExtra("newMessage", 0);
            PersonalFragment.this.initMessageNew(intent.getIntExtra("newMessageTotal", 0), intent.getIntExtra("newFirstChat", 0), intent.getIntExtra("newInteractive", 0), intent.getIntExtra("newNotice", 0));
        }
    };

    private void init() {
        UserModel userModel;
        this.baseRequest = new RetrofitRequest();
        this.gson = new Gson();
        this.permissionUtil = new PermissionUtil(this.mainActivity);
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.mainActivity, 3);
        this.mLayoutManager = gridLayoutManager;
        this.mRecyclerView.setLayoutManager(gridLayoutManager);
        ArrayList arrayList = new ArrayList();
        this.mList = arrayList;
        arrayList.add(new ActionModel(R.mipmap.ic_personal_message, LanguageUtil.getLanguageByKey(this.mainActivity, "messageTitle")));
        this.mList.add(new ActionModel(R.mipmap.ic_personal_collection, LanguageUtil.getLanguageByKey(this.mainActivity, "favoriteTitle")));
        this.mList.add(new ActionModel(R.mipmap.ic_personal_topic, LanguageUtil.getLanguageByKey(this.mainActivity, "topicTitle")));
        this.mList.add(new ActionModel(R.mipmap.ic_personal_interactive, LanguageUtil.getLanguageByKey(this.mainActivity, "interactiveTitle")));
        this.mList.add(new ActionModel(R.mipmap.ic_personal_like, LanguageUtil.getLanguageByKey(this.mainActivity, "like")));
        this.mList.add(new ActionModel(R.mipmap.ic_personal_evaluation, LanguageUtil.getLanguageByKey(this.mainActivity, "comment")));
        this.mList.add(new ActionModel(R.mipmap.ic_personal_black, LanguageUtil.getLanguageByKey(this.mainActivity, "blocking")));
        this.mList.add(new ActionModel(R.mipmap.ic_personal_play_record, LanguageUtil.getLanguageByKey(this.mainActivity, "playRecord")));
        ActionAdapter actionAdapter = new ActionAdapter(this.mainActivity, this.mList);
        this.mAdapter = actionAdapter;
        this.mRecyclerView.setAdapter(actionAdapter);
        try {
            userModel = (UserModel) this.gson.fromJson((String) SharedPreferencesUtil.get(this.mainActivity, Constans.UserModel), UserModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            userModel = null;
        }
        if (userModel != null) {
            LogUtil.e("SharedPreferencesUtil 读取到用户信息");
            initUserInfo(userModel);
        }
        registerReceiver();
    }

    private void initCameraPermission() {
        Disposable subscribe = new RxPermissions(this).requestEachCombined(Build.VERSION.SDK_INT >= 34 ? new String[]{"android.permission.CAMERA"} : Build.VERSION.SDK_INT >= 33 ? new String[]{"android.permission.CAMERA"} : new String[]{"android.permission.CAMERA", PermissionConfig.READ_EXTERNAL_STORAGE}).subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.PersonalFragment$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                PersonalFragment.this.m598x314886d((Permission) obj);
            }
        });
        if (this.mainActivity.mCompositeDisposable == null || this.mainActivity.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.mainActivity.mCompositeDisposable.add(subscribe);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initMessageNew(int i, int i2, int i3, int i4) {
        List<ActionModel> list = this.mList;
        if (i != 0) {
            list.get(0).setNew(true);
            this.mAdapter.notifyItemChanged(0, "");
        } else {
            list.get(0).setNew(false);
            this.mAdapter.notifyItemChanged(0, "");
        }
        List<ActionModel> list2 = this.mList;
        if (i3 != 0) {
            list2.get(3).setNew(true);
            this.mAdapter.notifyItemChanged(3, "");
        } else {
            list2.get(3).setNew(false);
            this.mAdapter.notifyItemChanged(3, "");
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initMessageRead(boolean z) {
        this.mList.get(0).setNew(z);
        this.mAdapter.notifyItemChanged(0, "");
    }

    private void initView(View view) {
        this.mLayoutHead = (LinearLayout) view.findViewById(R.id.layout_head);
        this.mImgScan = (ImageView) view.findViewById(R.id.personal_img_scan);
        this.mImgSetting = (ImageView) view.findViewById(R.id.personal_img_setting);
        this.mImgHead = (CircleImageView) view.findViewById(R.id.personal_img_head);
        this.mTxtName = (TextView) view.findViewById(R.id.personal_txt_name);
        this.mTxtConstellation = (TextView) view.findViewById(R.id.personal_txt_constellation);
        this.mTxtFriend = (TextView) view.findViewById(R.id.personal_txt_friend);
        this.mTxtFriendCount = (TextView) view.findViewById(R.id.personal_txt_friendCount);
        this.mTxtFans = (TextView) view.findViewById(R.id.personal_txt_fans);
        this.mTxtFansCount = (TextView) view.findViewById(R.id.personal_txt_fansCount);
        this.mTxtFollow = (TextView) view.findViewById(R.id.personal_txt_follow);
        this.mTxtFollowCount = (TextView) view.findViewById(R.id.personal_txt_followCount);
        this.mImgGender = (ImageView) view.findViewById(R.id.personal_img_gender);
        this.mImgArrow = (ImageView) view.findViewById(R.id.personal_img_arrow);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_recycler);
        this.mLayoutInfo = (LinearLayout) view.findViewById(R.id.personal_layout_info);
        this.mLayoutFriend = (LinearLayout) view.findViewById(R.id.personal_layout_friend);
        this.mLayoutFans = (LinearLayout) view.findViewById(R.id.personal_layout_fans);
        this.mLayoutFollow = (LinearLayout) view.findViewById(R.id.personal_layout_follow);
        this.mLayoutHead.setPadding(0, MeasureUtil.getStatusBarHeight(this.mainActivity), 0, MeasureUtil.dip2px(this.mainActivity, 15.0f));
        LanguageUtil.setText(this.mainActivity, this.mTxtFriend, "friendCount");
        LanguageUtil.setText(this.mainActivity, this.mTxtFans, "fansCount");
        LanguageUtil.setText(this.mainActivity, this.mTxtFollow, "following");
        this.isCreateView = true;
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_MESSAGE_NEW);
        intentFilter.addAction(Constans.RECEIVER_MESSAGE_READ);
        AppUtil.registerReceiver(this.mainActivity, this.mReceiver, intentFilter);
    }

    private void setListener() {
        this.mImgScan.setOnClickListener(this);
        this.mImgSetting.setOnClickListener(this);
        this.mImgHead.setOnClickListener(this);
        this.mLayoutInfo.setOnClickListener(this);
        this.mLayoutFriend.setOnClickListener(this);
        this.mLayoutFans.setOnClickListener(this);
        this.mLayoutFollow.setOnClickListener(this);
        this.mAdapter.setOnItemCLickListener(this);
    }

    private void unRegisterReceiver() {
        try {
            this.mainActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void initUserInfo(UserModel userModel) {
        if (this.isFirst && TextUtils.isEmpty(userModel.getName())) {
            this.isFirst = false;
            String str = (String) SharedPreferencesUtil.get(this.mainActivity, Constans.Token);
            Intent intent = new Intent(this.mainActivity, (Class<?>) InfoActivity.class);
            intent.putExtra(Constans.Token, str);
            this.mainActivity.startActivityForResult(intent, requestInfo);
        }
        if (this.isCreateView) {
            Glide.with((FragmentActivity) this.mainActivity).load(userModel.getHeadIcon()).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.fragment.PersonalFragment.1
                public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                    PersonalFragment.this.mImgHead.setImageDrawable(drawable);
                }

                @Override // com.bumptech.glide.request.target.Target
                public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                    onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
                }
            });
            this.mImgHead.setKing(userModel.getKing(), this.mainActivity, userModel.getGradeIcon());
            this.mTxtName.setText(userModel.getName());
            this.mTxtName.requestLayout();
            this.mTxtConstellation.setText(userModel.getConstellation());
            if (userModel.getGender() == 1) {
                this.mImgGender.setImageDrawable(ContextCompat.getDrawable(this.mainActivity, R.mipmap.ic_details_men));
            } else {
                int gender = userModel.getGender();
                ImageView imageView = this.mImgGender;
                if (gender == 2) {
                    imageView.setImageDrawable(ContextCompat.getDrawable(this.mainActivity, R.mipmap.ic_details_transgender));
                } else {
                    imageView.setImageDrawable(ContextCompat.getDrawable(this.mainActivity, R.mipmap.ic_details_women));
                }
            }
            this.mTxtFriendCount.setText(userModel.getFriendCount() + "");
            this.mTxtFansCount.setText(userModel.getFansCount() + "");
            this.mTxtFollowCount.setText(userModel.getFocusCount() + "");
            SharedPreferencesUtil.put(this.mainActivity, Constans.UserModel, this.gson.toJson(userModel));
            Message message = new Message();
            message.what = 1;
            message.obj = userModel.getHeadIcon();
            message.arg1 = userModel.getKing();
            this.mainActivity.mHandler.sendMessage(message);
        }
    }

    /* renamed from: lambda$initCameraPermission$0$com-hismith-hismithapp-fragment-PersonalFragment, reason: not valid java name */
    /* synthetic */ void m598x314886d(Permission permission) throws Exception {
        if (!permission.granted) {
            if (permission.shouldShowRequestPermissionRationale) {
                return;
            }
            ToastUtil.showShortToast(this.mainActivity, LanguageUtil.getLanguageByKey(this.mainActivity, "cameraPermission"));
            return;
        }
        if (Build.VERSION.SDK_INT >= 34) {
            LogUtil.e("android 14 权限允许");
        } else if (Build.VERSION.SDK_INT >= 33) {
            LogUtil.e("android 13 权限允许");
        }
        startActivity(new Intent(this.mainActivity, (Class<?>) QRCodeActivity.class));
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.personal_img_scan) {
            this.permissionUtil.initCameraPermission(new PermissionUtil.OnGrantedListener() { // from class: com.hismith.hismithapp.fragment.PersonalFragment.2
                @Override // com.hismith.hismithapp.utils.PermissionUtil.OnGrantedListener
                public void onGranted() {
                    PersonalFragment.this.startActivity(new Intent(PersonalFragment.this.mainActivity, (Class<?>) QRCodeActivity.class));
                }
            });
            return;
        }
        if (id == R.id.personal_img_setting) {
            startActivity(new Intent(this.mainActivity, (Class<?>) SettingActivity.class));
            return;
        }
        if (id == R.id.personal_layout_friend) {
            startActivity(new Intent(this.mainActivity, (Class<?>) FriendActivity.class));
            return;
        }
        if (id == R.id.personal_layout_fans) {
            Intent intent = new Intent(this.mainActivity, (Class<?>) FansFocusActivity.class);
            intent.putExtra("fansMode", 1);
            startActivity(intent);
        } else if (id == R.id.personal_layout_follow) {
            Intent intent2 = new Intent(this.mainActivity, (Class<?>) FansFocusActivity.class);
            intent2.putExtra("fansMode", 0);
            startActivity(intent2);
        } else if (id == R.id.personal_img_head || id == R.id.personal_layout_info) {
            startActivity(new Intent(this.mainActivity, (Class<?>) PersonalDetailsActivity2.class));
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mainActivity = (MainActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal, (ViewGroup) null);
        initView(inflate);
        init();
        setListener();
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        unRegisterReceiver();
    }

    @Override // com.hismith.hismithapp.adapter.ActionAdapter.OnItemClickListener
    public void onItemClick(int i) {
        if (i == 0) {
            Intent intent = new Intent(this.mainActivity, (Class<?>) PersonalMessageActivity.class);
            intent.putExtra("extra_position", i);
            startActivity(intent);
        } else {
            if (i == 6) {
                startActivity(new Intent(this.mainActivity, (Class<?>) BlackListActivity.class));
                return;
            }
            if (i == 1) {
                startActivity(new Intent(this.mainActivity, (Class<?>) PersonalCollectionActivity.class));
            } else {
                if (i == 7) {
                    startActivity(new Intent(this.mainActivity, (Class<?>) PlayStatistics2Activity.class));
                    return;
                }
                Intent intent2 = new Intent(this.mainActivity, (Class<?>) PersonalActionActivity.class);
                intent2.putExtra("extra_position", i - 2);
                startActivity(intent2);
            }
        }
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageScrollStateChanged(int i) {
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageScrolled(int i, float f, int i2) {
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageSelected(int i) {
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
    }
}
