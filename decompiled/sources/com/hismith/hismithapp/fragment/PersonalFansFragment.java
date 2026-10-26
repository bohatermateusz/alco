package com.hismith.hismithapp.fragment;

import android.content.ContentValues;
import android.content.Intent;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
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
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalFansFragment extends LazyLoadFragment {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbWrite;
    private FansAdapter mAdapter;
    private List<FansFocusModel.User> mFanList;
    private FrameLayout mLayout;
    private LinearLayoutManager mLinearLayoutManager;
    private SlideRecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private int page = 1;
    private int totalPage = 1;

    static /* synthetic */ int access$008(PersonalFansFragment personalFansFragment) {
        int i = personalFansFragment.page;
        personalFansFragment.page = i + 1;
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
    public void follow(final Integer num) {
        Observable<FollowModel> observeOn = this.baseRequest.getApi().follow((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), Integer.valueOf(this.mFanList.get(num.intValue()).getUserId()), Integer.valueOf(this.mFanList.get(num.intValue()).getIsField().intValue() == 0 ? 1 : 0)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<FollowModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.PersonalFansFragment.6
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ((FansFocusModel.User) PersonalFansFragment.this.mFanList.get(num.intValue())).setFollowStatus(0);
                PersonalFansFragment.this.mAdapter.notifyItemChanged(num.intValue(), "");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((FansFocusModel.User) PersonalFansFragment.this.mFanList.get(num.intValue())).setFollowStatus(0);
                PersonalFansFragment.this.mAdapter.notifyItemChanged(num.intValue(), "");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FollowModel followModel) {
                if (followModel.getSuccess() == 1) {
                    ((FansFocusModel.User) PersonalFansFragment.this.mFanList.get(num.intValue())).setIsField(followModel.getFollow());
                    if (followModel.getFollow().intValue() == 1) {
                        if (MyApplication.getApplication().getmUserModel() != null) {
                            MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() + 1);
                            PersonalFansFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                        }
                        ToastUtil.showShortToastCenter(PersonalFansFragment.this.baseActivity, LanguageUtil.getLanguageByKey(PersonalFansFragment.this.baseActivity, "followedMsg"));
                        return;
                    }
                    if (MyApplication.getApplication().getmUserModel() != null) {
                        MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() - 1);
                        PersonalFansFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                    ToastUtil.showShortToastCenter(PersonalFansFragment.this.baseActivity, LanguageUtil.getLanguageByKey(PersonalFansFragment.this.baseActivity, "unfollowedMsg"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getFansFocus(final int i) {
        Observable<FansFocusModel> observeOn = this.baseRequest.getApi().getFansFocus((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), "fans", Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<FansFocusModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.PersonalFansFragment.5
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalFansFragment.this.mRefreshLayout.finishRefresh();
                PersonalFansFragment.this.mRefreshLayout.finishLoadMore();
                NetErrorDialog.dissErrorDialog(PersonalFansFragment.this.baseActivity, PersonalFansFragment.this.mLayout);
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalFansFragment.this.mRefreshLayout.finishRefresh();
                PersonalFansFragment.this.mRefreshLayout.finishLoadMore();
                if (i == 1) {
                    NetErrorDialog.showErrorDialog(PersonalFansFragment.this.baseActivity, PersonalFansFragment.this.mLayout, new NetErrorDialog.OnResetClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalFansFragment.5.1
                        @Override // com.hismith.hismithapp.dialog.NetErrorDialog.OnResetClickListener
                        public void onResetClick() {
                            PersonalFansFragment.this.mRefreshLayout.autoRefresh();
                        }
                    });
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FansFocusModel fansFocusModel) {
                try {
                    if (fansFocusModel.getSuccess() == 1) {
                        PersonalFansFragment.this.page = fansFocusModel.getPage();
                        PersonalFansFragment.this.totalPage = fansFocusModel.getTotalPage();
                        int totalCount = fansFocusModel.getTotalCount();
                        if (PersonalFansFragment.this.page <= 1) {
                            PersonalFansFragment.this.mFanList.clear();
                        }
                        if (fansFocusModel.getUsers() != null) {
                            if (totalCount < fansFocusModel.getUsers().length) {
                                totalCount = fansFocusModel.getUsers().length;
                            }
                            PersonalFansFragment.this.mFanList.addAll(Arrays.asList(fansFocusModel.getUsers()));
                            PersonalFansFragment.this.insertDBMessageUser(fansFocusModel);
                        }
                        PersonalFansFragment.this.mAdapter.notifyDataSetChanged();
                        if (MyApplication.getApplication().getmUserModel() != null) {
                            MyApplication.getApplication().getmUserModel().setFansCount(totalCount);
                            PersonalFansFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
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
        DatabaseHelper databaseHelper = new DatabaseHelper(this.baseActivity);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mFanList = new ArrayList();
        FansAdapter fansAdapter = new FansAdapter(this.baseActivity, this.mFanList);
        this.mAdapter = fansAdapter;
        this.mRecyclerView.setAdapter(fansAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mLayout = (FrameLayout) view.findViewById(R.id.fans_layout);
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.fans_refreshLayout);
        this.mRecyclerView = (SlideRecyclerView) view.findViewById(R.id.fans_recycler_view);
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
        Observable<FollowModel> observeOn = this.baseRequest.getApi().removeFan((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), Integer.valueOf(this.mFanList.get(i).getUserId()), 0, 1).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<FollowModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.fragment.PersonalFansFragment.7
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FollowModel followModel) {
                if (followModel.getSuccess() == 1) {
                    PersonalFansFragment.this.mFanList.remove(i);
                    PersonalFansFragment.this.mAdapter.notifyItemRemoved(i);
                    if (i < PersonalFansFragment.this.mFanList.size()) {
                        PersonalFansFragment.this.mAdapter.notifyItemRangeChanged(i, PersonalFansFragment.this.mFanList.size() - i);
                    }
                    if (MyApplication.getApplication().getmUserModel() != null) {
                        MyApplication.getApplication().getmUserModel().setFansCount(MyApplication.getApplication().getmUserModel().getFansCount() - 1);
                        PersonalFansFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                }
            }
        });
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalFansFragment.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = PersonalFansFragment.this.page;
                int i2 = PersonalFansFragment.this.totalPage;
                PersonalFansFragment personalFansFragment = PersonalFansFragment.this;
                if (i >= i2) {
                    personalFansFragment.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                PersonalFansFragment.access$008(personalFansFragment);
                PersonalFansFragment personalFansFragment2 = PersonalFansFragment.this;
                personalFansFragment2.getFansFocus(personalFansFragment2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PersonalFansFragment.this.page = 1;
                PersonalFansFragment personalFansFragment = PersonalFansFragment.this;
                personalFansFragment.getFansFocus(personalFansFragment.page);
            }
        });
        this.mAdapter.setOnItemFollowClickListener(new FansAdapter.OnItemFollowClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalFansFragment.2
            @Override // com.hismith.hismithapp.adapter.FansAdapter.OnItemFollowClickListener
            public void onItemFollow(int i) {
                PersonalFansFragment.this.follow(Integer.valueOf(i));
            }
        });
        this.mAdapter.setOnItemCLickListener(new FansAdapter.OnItemCLickListener() { // from class: com.hismith.hismithapp.fragment.PersonalFansFragment.3
            @Override // com.hismith.hismithapp.adapter.FansAdapter.OnItemCLickListener
            public void onItemClick(int i) {
                IntentUtil.startUserInfoActivity(PersonalFansFragment.this.baseActivity, ((FansFocusModel.User) PersonalFansFragment.this.mFanList.get(i)).getUserId());
            }
        });
        this.mAdapter.setOnItemRemoveClickListener(new FansAdapter.OnItemRemoveClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalFansFragment.4
            @Override // com.hismith.hismithapp.adapter.FansAdapter.OnItemRemoveClickListener
            public void onItemRemove(int i) {
                PersonalFansFragment.this.removeFan(i);
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_fans, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        closeDB();
    }
}
