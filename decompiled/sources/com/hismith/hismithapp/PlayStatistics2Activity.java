package com.hismith.hismithapp;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.facebook.appevents.iap.InAppPurchaseConstants;
import com.hismith.hismithapp.adapter.PlayStatisticsAdapter;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.PlayLogsListModel;
import com.hismith.hismithapp.model.PlayLogsModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import kotlin.Metadata;
import kotlin.collections.CollectionsKt;
import kotlin.jvm.internal.Intrinsics;

/* compiled from: PlayStatistics2Activity.kt */
@Metadata(d1 = {"\u0000X\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0002\b\u0003\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0010\u000e\n\u0000\n\u0002\u0010\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0005\n\u0002\u0010\b\n\u0000\u0018\u00002\u00020\u0001B\u0007¢\u0006\u0004\b\u0002\u0010\u0003J\u0012\u0010\u0016\u001a\u00020\u00172\b\u0010\u0018\u001a\u0004\u0018\u00010\u0019H\u0014J\u0006\u0010\u001a\u001a\u00020\u0017J\u0006\u0010\u001b\u001a\u00020\u0017J\u0006\u0010\u001c\u001a\u00020\u0017J\u000e\u0010\u001d\u001a\u00020\u00172\u0006\u0010\u001e\u001a\u00020\u001fR\u000e\u0010\u0004\u001a\u00020\u0005X\u0082.¢\u0006\u0002\n\u0000R\u000e\u0010\u0006\u001a\u00020\u0007X\u0082.¢\u0006\u0002\n\u0000R\u000e\u0010\b\u001a\u00020\tX\u0082.¢\u0006\u0002\n\u0000R\u000e\u0010\n\u001a\u00020\u000bX\u0082.¢\u0006\u0002\n\u0000R\u000e\u0010\f\u001a\u00020\rX\u0082\u000e¢\u0006\u0002\n\u0000R\u001e\u0010\u000e\u001a\u0012\u0012\u0004\u0012\u00020\u00100\u000fj\b\u0012\u0004\u0012\u00020\u0010`\u0011X\u0082.¢\u0006\u0002\n\u0000R\u000e\u0010\u0012\u001a\u00020\u0013X\u0082.¢\u0006\u0002\n\u0000R\u000e\u0010\u0014\u001a\u00020\u0015X\u0082\u000e¢\u0006\u0002\n\u0000¨\u0006 "}, d2 = {"Lcom/hismith/hismithapp/PlayStatistics2Activity;", "Lcom/hismith/hismithapp/BaseActivity;", "<init>", "()V", "title_img_back", "Landroid/widget/ImageView;", "title_txt_name", "Landroid/widget/TextView;", "refresh_layout", "Lcom/scwang/smart/refresh/layout/api/RefreshLayout;", "recycler_data", "Landroidx/recyclerview/widget/RecyclerView;", "baseRequest", "Lcom/hismith/hismithapp/http/RetrofitRequest;", "datas", "Ljava/util/ArrayList;", "Lcom/hismith/hismithapp/model/PlayLogsModel;", "Lkotlin/collections/ArrayList;", "mAdapter", "Lcom/hismith/hismithapp/adapter/PlayStatisticsAdapter;", "requestId", "", "onCreate", "", "savedInstanceState", "Landroid/os/Bundle;", "initView", "init", InAppPurchaseConstants.METHOD_SET_LISTENER, "playLogs", "mNew", "", "app"}, k = 1, mv = {2, 3, 0}, xi = 48)
/* loaded from: classes3.dex */
public final class PlayStatistics2Activity extends BaseActivity {
    private ArrayList<PlayLogsModel> datas;
    private PlayStatisticsAdapter mAdapter;
    private RecyclerView recycler_data;
    private RefreshLayout refresh_layout;
    private ImageView title_img_back;
    private TextView title_txt_name;
    private RetrofitRequest baseRequest = new RetrofitRequest();
    private String requestId = "";

    public final void init() {
        RecyclerView recyclerView = this.recycler_data;
        PlayStatisticsAdapter playStatisticsAdapter = null;
        if (recyclerView == null) {
            Intrinsics.throwUninitializedPropertyAccessException("recycler_data");
            recyclerView = null;
        }
        PlayStatistics2Activity playStatistics2Activity = this;
        recyclerView.setLayoutManager(new LinearLayoutManager(playStatistics2Activity));
        RecyclerView recyclerView2 = this.recycler_data;
        if (recyclerView2 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("recycler_data");
            recyclerView2 = null;
        }
        recyclerView2.addItemDecoration(new DividerItemDecoration(playStatistics2Activity, 1));
        this.datas = new ArrayList<>();
        ArrayList<PlayLogsModel> arrayList = this.datas;
        if (arrayList == null) {
            Intrinsics.throwUninitializedPropertyAccessException("datas");
            arrayList = null;
        }
        this.mAdapter = new PlayStatisticsAdapter(playStatistics2Activity, arrayList);
        RecyclerView recyclerView3 = this.recycler_data;
        if (recyclerView3 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("recycler_data");
            recyclerView3 = null;
        }
        PlayStatisticsAdapter playStatisticsAdapter2 = this.mAdapter;
        if (playStatisticsAdapter2 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("mAdapter");
        } else {
            playStatisticsAdapter = playStatisticsAdapter2;
        }
        recyclerView3.setAdapter(playStatisticsAdapter);
    }

    public final void initView() {
        View findViewById = findViewById(R.id.title_img_back);
        Intrinsics.checkNotNullExpressionValue(findViewById, "findViewById(...)");
        this.title_img_back = (ImageView) findViewById;
        View findViewById2 = findViewById(R.id.title_txt_name);
        Intrinsics.checkNotNullExpressionValue(findViewById2, "findViewById(...)");
        this.title_txt_name = (TextView) findViewById2;
        this.refresh_layout = (RefreshLayout) findViewById(R.id.refresh_layout);
        View findViewById3 = findViewById(R.id.recycler_data);
        Intrinsics.checkNotNullExpressionValue(findViewById3, "findViewById(...)");
        this.recycler_data = (RecyclerView) findViewById3;
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_play_statistics);
        initView();
        init();
        setListener();
        PlayStatistics2Activity playStatistics2Activity = this;
        TextView textView = this.title_txt_name;
        if (textView == null) {
            Intrinsics.throwUninitializedPropertyAccessException("title_txt_name");
            textView = null;
        }
        LanguageUtil.setText(playStatistics2Activity, textView, "playRecord");
    }

    public final void playLogs(int mNew) {
        this.baseRequest.getApi().playLogs(SharedPreferencesUtil.get(this, Constans.Token).toString(), mNew, this.requestId).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayLogsListModel>() { // from class: com.hismith.hismithapp.PlayStatistics2Activity$playLogs$1
            {
                super(PlayStatistics2Activity.this, null, false);
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                RefreshLayout refreshLayout;
                RefreshLayout refreshLayout2;
                super.onComplete();
                refreshLayout = PlayStatistics2Activity.this.refresh_layout;
                RefreshLayout refreshLayout3 = null;
                if (refreshLayout == null) {
                    Intrinsics.throwUninitializedPropertyAccessException("refresh_layout");
                    refreshLayout = null;
                }
                refreshLayout.finishRefresh();
                refreshLayout2 = PlayStatistics2Activity.this.refresh_layout;
                if (refreshLayout2 == null) {
                    Intrinsics.throwUninitializedPropertyAccessException("refresh_layout");
                } else {
                    refreshLayout3 = refreshLayout2;
                }
                refreshLayout3.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable e) {
                RefreshLayout refreshLayout;
                RefreshLayout refreshLayout2;
                Intrinsics.checkNotNullParameter(e, "e");
                super.onError(e);
                refreshLayout = PlayStatistics2Activity.this.refresh_layout;
                RefreshLayout refreshLayout3 = null;
                if (refreshLayout == null) {
                    Intrinsics.throwUninitializedPropertyAccessException("refresh_layout");
                    refreshLayout = null;
                }
                refreshLayout.finishRefresh();
                refreshLayout2 = PlayStatistics2Activity.this.refresh_layout;
                if (refreshLayout2 == null) {
                    Intrinsics.throwUninitializedPropertyAccessException("refresh_layout");
                } else {
                    refreshLayout3 = refreshLayout2;
                }
                refreshLayout3.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayLogsListModel t) {
                String str;
                ArrayList arrayList;
                PlayStatisticsAdapter playStatisticsAdapter;
                ArrayList arrayList2;
                if (t == null || t.getSuccess() != 1) {
                    return;
                }
                str = PlayStatistics2Activity.this.requestId;
                PlayStatisticsAdapter playStatisticsAdapter2 = null;
                if (TextUtils.isEmpty(str)) {
                    arrayList2 = PlayStatistics2Activity.this.datas;
                    if (arrayList2 == null) {
                        Intrinsics.throwUninitializedPropertyAccessException("datas");
                        arrayList2 = null;
                    }
                    arrayList2.clear();
                }
                if (!TextUtils.isEmpty(t.getRequestId())) {
                    PlayStatistics2Activity playStatistics2Activity = PlayStatistics2Activity.this;
                    String requestId = t.getRequestId();
                    Intrinsics.checkNotNullExpressionValue(requestId, "getRequestId(...)");
                    playStatistics2Activity.requestId = requestId;
                }
                PlayLogsModel[] playLogs = t.getPlayLogs();
                if (playLogs == null || playLogs.length == 0) {
                    return;
                }
                arrayList = PlayStatistics2Activity.this.datas;
                if (arrayList == null) {
                    Intrinsics.throwUninitializedPropertyAccessException("datas");
                    arrayList = null;
                }
                PlayLogsModel[] playLogs2 = t.getPlayLogs();
                Intrinsics.checkNotNullExpressionValue(playLogs2, "getPlayLogs(...)");
                CollectionsKt.addAll(arrayList, playLogs2);
                playStatisticsAdapter = PlayStatistics2Activity.this.mAdapter;
                if (playStatisticsAdapter == null) {
                    Intrinsics.throwUninitializedPropertyAccessException("mAdapter");
                } else {
                    playStatisticsAdapter2 = playStatisticsAdapter;
                }
                playStatisticsAdapter2.notifyDataSetChanged();
            }
        });
    }

    public final void setListener() {
        RefreshLayout refreshLayout = this.refresh_layout;
        ImageView imageView = null;
        if (refreshLayout == null) {
            Intrinsics.throwUninitializedPropertyAccessException("refresh_layout");
            refreshLayout = null;
        }
        refreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.PlayStatistics2Activity$setListener$1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout2) {
                Intrinsics.checkNotNullParameter(refreshLayout2, "refreshLayout");
                PlayStatistics2Activity.this.playLogs(0);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout2) {
                Intrinsics.checkNotNullParameter(refreshLayout2, "refreshLayout");
                PlayStatistics2Activity.this.requestId = "";
                PlayStatistics2Activity.this.playLogs(1);
            }
        });
        RefreshLayout refreshLayout2 = this.refresh_layout;
        if (refreshLayout2 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("refresh_layout");
            refreshLayout2 = null;
        }
        refreshLayout2.autoRefresh();
        ImageView imageView2 = this.title_img_back;
        if (imageView2 == null) {
            Intrinsics.throwUninitializedPropertyAccessException("title_img_back");
        } else {
            imageView = imageView2;
        }
        imageView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayStatistics2Activity$$ExternalSyntheticLambda0
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayStatistics2Activity.this.finish();
            }
        });
    }
}
