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
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.FocusAdapter;
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
public class PersonalFocusFragment extends LazyLoadFragment {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbWrite;
    private FocusAdapter mAdapter;
    private List<FansFocusModel.User> mFocusList;
    private FrameLayout mLayout;
    private LinearLayoutManager mLinearLayoutManager;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private int page = 1;
    private int totalPage = 1;

    static /* synthetic */ int access$008(PersonalFocusFragment personalFocusFragment) {
        int i = personalFocusFragment.page;
        personalFocusFragment.page = i + 1;
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
        Observable<FollowModel> observeOn = this.baseRequest.getApi().follow((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), Integer.valueOf(this.mFocusList.get(num.intValue()).getUserId()), Integer.valueOf(this.mFocusList.get(num.intValue()).IsFollow() == 0 ? 1 : 0)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<FollowModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.PersonalFocusFragment.5
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ((FansFocusModel.User) PersonalFocusFragment.this.mFocusList.get(num.intValue())).setFollowStatus(0);
                PersonalFocusFragment.this.mAdapter.notifyItemChanged(num.intValue(), "");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((FansFocusModel.User) PersonalFocusFragment.this.mFocusList.get(num.intValue())).setFollowStatus(0);
                PersonalFocusFragment.this.mAdapter.notifyItemChanged(num.intValue(), "");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FollowModel followModel) {
                if (followModel.getSuccess() == 1) {
                    ((FansFocusModel.User) PersonalFocusFragment.this.mFocusList.get(num.intValue())).setFollow(followModel.getFollow().intValue());
                    if (followModel.getFollow().intValue() == 1) {
                        if (MyApplication.getApplication().getmUserModel() != null) {
                            MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() + 1);
                            PersonalFocusFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                        }
                        ToastUtil.showShortToastCenter(PersonalFocusFragment.this.baseActivity, LanguageUtil.getLanguageByKey(PersonalFocusFragment.this.baseActivity, "followedMsg"));
                        return;
                    }
                    if (MyApplication.getApplication().getmUserModel() != null) {
                        MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() - 1);
                        PersonalFocusFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                    ToastUtil.showShortToastCenter(PersonalFocusFragment.this.baseActivity, LanguageUtil.getLanguageByKey(PersonalFocusFragment.this.baseActivity, "unfollowedMsg"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getFansFocus(final int i) {
        Observable<FansFocusModel> observeOn = this.baseRequest.getApi().getFansFocus((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), "focus", Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<FansFocusModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.PersonalFocusFragment.4
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalFocusFragment.this.mRefreshLayout.finishRefresh();
                PersonalFocusFragment.this.mRefreshLayout.finishLoadMore();
                NetErrorDialog.dissErrorDialog(PersonalFocusFragment.this.baseActivity, PersonalFocusFragment.this.mLayout);
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalFocusFragment.this.mRefreshLayout.finishRefresh();
                PersonalFocusFragment.this.mRefreshLayout.finishLoadMore();
                if (i == 1) {
                    NetErrorDialog.showErrorDialog(PersonalFocusFragment.this.baseActivity, PersonalFocusFragment.this.mLayout, new NetErrorDialog.OnResetClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalFocusFragment.4.1
                        @Override // com.hismith.hismithapp.dialog.NetErrorDialog.OnResetClickListener
                        public void onResetClick() {
                            PersonalFocusFragment.this.mRefreshLayout.autoRefresh();
                        }
                    });
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FansFocusModel fansFocusModel) {
                if (fansFocusModel.getSuccess() == 1) {
                    PersonalFocusFragment.this.page = fansFocusModel.getPage();
                    PersonalFocusFragment.this.totalPage = fansFocusModel.getTotalPage();
                    int totalCount = fansFocusModel.getTotalCount();
                    if (PersonalFocusFragment.this.page <= 1) {
                        PersonalFocusFragment.this.mFocusList.clear();
                    }
                    if (fansFocusModel.getUsers() != null) {
                        if (totalCount < fansFocusModel.getUsers().length) {
                            totalCount = fansFocusModel.getUsers().length;
                        }
                        for (FansFocusModel.User user : Arrays.asList(fansFocusModel.getUsers())) {
                            user.setFollow(1);
                            PersonalFocusFragment.this.mFocusList.add(user);
                        }
                        PersonalFocusFragment.this.insertDBMessageUser(fansFocusModel);
                    }
                    PersonalFocusFragment.this.mAdapter.notifyDataSetChanged();
                    if (MyApplication.getApplication().getmUserModel() != null) {
                        MyApplication.getApplication().getmUserModel().setFocusCount(totalCount);
                        PersonalFocusFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
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
        this.mFocusList = new ArrayList();
        FocusAdapter focusAdapter = new FocusAdapter(this.baseActivity, this.mFocusList);
        this.mAdapter = focusAdapter;
        this.mRecyclerView.setAdapter(focusAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mLayout = (FrameLayout) view.findViewById(R.id.focus_layout);
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.focus_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.focus_recycler_view);
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

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalFocusFragment.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = PersonalFocusFragment.this.page;
                int i2 = PersonalFocusFragment.this.totalPage;
                PersonalFocusFragment personalFocusFragment = PersonalFocusFragment.this;
                if (i >= i2) {
                    personalFocusFragment.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                PersonalFocusFragment.access$008(personalFocusFragment);
                PersonalFocusFragment personalFocusFragment2 = PersonalFocusFragment.this;
                personalFocusFragment2.getFansFocus(personalFocusFragment2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PersonalFocusFragment.this.page = 1;
                PersonalFocusFragment personalFocusFragment = PersonalFocusFragment.this;
                personalFocusFragment.getFansFocus(personalFocusFragment.page);
            }
        });
        this.mAdapter.setOnItemCLickListener(new FocusAdapter.OnItemCLickListener() { // from class: com.hismith.hismithapp.fragment.PersonalFocusFragment.2
            @Override // com.hismith.hismithapp.adapter.FocusAdapter.OnItemCLickListener
            public void onItemClick(int i) {
                IntentUtil.startUserInfoActivity(PersonalFocusFragment.this.baseActivity, ((FansFocusModel.User) PersonalFocusFragment.this.mFocusList.get(i)).getUserId());
            }
        });
        this.mAdapter.setOnItemFollowClickListener(new FocusAdapter.OnItemFollowClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalFocusFragment.3
            @Override // com.hismith.hismithapp.adapter.FocusAdapter.OnItemFollowClickListener
            public void onItemFollow(int i) {
                PersonalFocusFragment.this.follow(Integer.valueOf(i));
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
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_focus, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        closeDB();
    }
}
