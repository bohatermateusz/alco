package com.hismith.hismithapp;

import android.content.ContentValues;
import android.content.Intent;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.hismith.hismithapp.adapter.FansAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.dialog.NetErrorDialog;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.FansFocusModel;
import com.hismith.hismithapp.model.FollowModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.IntentUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class FriendActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbWrite;
    private FansAdapter mAdapter;
    private List<FansFocusModel.User> mFanList;
    private FrameLayout mLayout;
    private LinearLayoutManager mLinearLayoutManager;
    private SlideRecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private ImageView mTitleBack;
    private ImageView mTitleImgAdd;
    private TextView mTitleName;
    private int page = 1;
    private int totalPage = 1;

    static /* synthetic */ int access$008(FriendActivity friendActivity) {
        int i = friendActivity.page;
        friendActivity.page = i + 1;
        return i;
    }

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
    public void getFansFocus(final int i) {
        this.baseRequest.getApi().getFansFocus((String) SharedPreferencesUtil.get(this, Constans.Token), "friend", Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FansFocusModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.FriendActivity.4
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                FriendActivity.this.mRefreshLayout.finishLoadMore();
                FriendActivity.this.mRefreshLayout.finishRefresh();
                FriendActivity friendActivity = FriendActivity.this;
                NetErrorDialog.dissErrorDialog(friendActivity, friendActivity.mLayout);
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                FriendActivity.this.mRefreshLayout.finishLoadMore();
                FriendActivity.this.mRefreshLayout.finishRefresh();
                if (i == 1) {
                    FriendActivity friendActivity = FriendActivity.this;
                    NetErrorDialog.showErrorDialog(friendActivity, friendActivity.mLayout, new NetErrorDialog.OnResetClickListener() { // from class: com.hismith.hismithapp.FriendActivity.4.1
                        @Override // com.hismith.hismithapp.dialog.NetErrorDialog.OnResetClickListener
                        public void onResetClick() {
                            FriendActivity.this.mRefreshLayout.autoRefresh();
                        }
                    });
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FansFocusModel fansFocusModel) {
                try {
                    if (fansFocusModel.getSuccess() == 1) {
                        FriendActivity.this.page = fansFocusModel.getPage();
                        FriendActivity.this.totalPage = fansFocusModel.getTotalPage();
                        int totalCount = fansFocusModel.getTotalCount();
                        if (FriendActivity.this.page <= 1) {
                            FriendActivity.this.mFanList.clear();
                        }
                        if (fansFocusModel.getUsers() != null) {
                            if (totalCount < fansFocusModel.getUsers().length) {
                                totalCount = fansFocusModel.getUsers().length;
                            }
                            FriendActivity.this.mFanList.addAll(Arrays.asList(fansFocusModel.getUsers()));
                            FriendActivity.this.insertDBMessageUser(fansFocusModel);
                        }
                        FriendActivity.this.mAdapter.notifyDataSetChanged();
                        if (MyApplication.getApplication().getmUserModel() != null) {
                            MyApplication.getApplication().getmUserModel().setFriendCount(totalCount);
                            FriendActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        DatabaseHelper databaseHelper = new DatabaseHelper(this);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mFanList = new ArrayList();
        FansAdapter fansAdapter = new FansAdapter(this, this.mFanList, false);
        this.mAdapter = fansAdapter;
        this.mRecyclerView.setAdapter(fansAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView() {
        this.mLayout = (FrameLayout) findViewById(R.id.friend_layout);
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleImgAdd = (ImageView) findViewById(R.id.title_img_add);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.friend_refreshLayout);
        this.mRecyclerView = (SlideRecyclerView) findViewById(R.id.friend_recycler_view);
        LanguageUtil.setText(this, this.mTitleName, "friendCount");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessageUser(FansFocusModel fansFocusModel) {
        for (int i = 0; i < fansFocusModel.getUsers().length; i++) {
            try {
                FansFocusModel.User user = fansFocusModel.getUsers()[i];
                ContentValues contentValues = new ContentValues();
                contentValues.put("userId", Integer.valueOf(user.getUserId()));
                contentValues.put("king", user.getKing());
                contentValues.put("gradeIcon", user.getGradeIcon());
                contentValues.put("name", user.getName());
                contentValues.put("gender", user.getGender());
                contentValues.put("headIcon", user.getHeadIcon());
                this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_USER, null, contentValues);
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void removeFan(final int i) {
        this.baseRequest.getApi().removeFan((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.mFanList.get(i).getUserId()), 0, 1).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FollowModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.FriendActivity.5
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FollowModel followModel) {
                if (followModel.getSuccess() == 1) {
                    FriendActivity.this.mFanList.remove(i);
                    FriendActivity.this.mAdapter.notifyItemRemoved(i);
                    if (i < FriendActivity.this.mFanList.size()) {
                        FriendActivity.this.mAdapter.notifyItemRangeChanged(i, FriendActivity.this.mFanList.size() - i);
                    }
                    if (MyApplication.getApplication().getmUserModel() != null) {
                        MyApplication.getApplication().getmUserModel().setFriendCount(MyApplication.getApplication().getmUserModel().getFriendCount() - 1);
                        FriendActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                }
            }
        });
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mTitleImgAdd.setOnClickListener(this);
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.FriendActivity.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = FriendActivity.this.page;
                int i2 = FriendActivity.this.totalPage;
                FriendActivity friendActivity = FriendActivity.this;
                if (i >= i2) {
                    friendActivity.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                FriendActivity.access$008(friendActivity);
                FriendActivity friendActivity2 = FriendActivity.this;
                friendActivity2.getFansFocus(friendActivity2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                FriendActivity.this.page = 1;
                FriendActivity friendActivity = FriendActivity.this;
                friendActivity.getFansFocus(friendActivity.page);
            }
        });
        this.mAdapter.setOnItemCLickListener(new FansAdapter.OnItemCLickListener() { // from class: com.hismith.hismithapp.FriendActivity.2
            @Override // com.hismith.hismithapp.adapter.FansAdapter.OnItemCLickListener
            public void onItemClick(int i) {
                FriendActivity friendActivity = FriendActivity.this;
                IntentUtil.startUserInfoActivity(friendActivity, ((FansFocusModel.User) friendActivity.mFanList.get(i)).getUserId());
            }
        });
        this.mAdapter.setOnItemRemoveClickListener(new FansAdapter.OnItemRemoveClickListener() { // from class: com.hismith.hismithapp.FriendActivity.3
            @Override // com.hismith.hismithapp.adapter.FansAdapter.OnItemRemoveClickListener
            public void onItemRemove(int i) {
                FriendActivity.this.removeFan(i);
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
        } else if (id == R.id.title_img_add) {
            startActivity(new Intent(this, (Class<?>) AddFriendActivity.class));
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_friend);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        closeDB();
    }
}
