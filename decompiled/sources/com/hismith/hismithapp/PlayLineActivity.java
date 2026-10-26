package com.hismith.hismithapp;

import android.app.Dialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import androidx.exifinterface.media.ExifInterface;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.facebook.internal.NativeProtocol;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.hismith.hismithapp.adapter.LineRoomAdapter;
import com.hismith.hismithapp.adapter.LineRoomMessageAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.LineUserDialog;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.socket.PlayLineSocketModel;
import com.hismith.hismithapp.model.socket.SocketRoomModel;
import com.hismith.hismithapp.model.socket.SocketUserModel;
import com.hismith.hismithapp.popup.MoreAttachPopup2;
import com.hismith.hismithapp.receiver.NetBroadcastReceiver;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BasePopupView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PlayLineActivity extends BaseActivity implements View.OnClickListener, LineRoomAdapter.OnItemClickListener, NetBroadcastReceiver.NetChangeListener {
    private BasePopupView loadPopupView;
    private LineRoomAdapter mAdapter;
    private BasePopupView mCreateTipPupup;
    private EditText mEditMessage;
    private ImageView mImgNetworkError;
    private ImageView mImgPeopleBg;
    private LinearLayout mLayoutLoading;
    private GridLayoutManager mLayoutManager;
    private LineUserDialog mLineUserDialog;
    private NetBroadcastReceiver mNetBroadcastReceiver;
    private RecyclerView mRecyclerView;
    private RecyclerView mRecyclerWorldMessage;
    private SmartRefreshLayout mRefreshLayout;
    private List<SocketRoomModel> mRoomList;
    private ImageView mTitleBack;
    private FrameLayout mTitleMore;
    private TextView mTitleTxtName;
    private TextView mTvLoading;
    private TextView mTxtNetworkError;
    private TextView mTxtPeoples;
    private List<SocketUserModel> mUserList;
    private View mViewNetworkError;
    private LineRoomMessageAdapter mWorldMessageAdapter;
    private LinearLayoutManager mWorldMessageLayoutManager;
    private List<PlayLineSocketModel> mWorldMessages;
    private MoreAttachPopup2 moreAttachPopup2;
    private int mLastNetStatus = -1;
    private int limitIndex = 0;
    private int isPublic = 1;
    private boolean isLoading = false;
    private final int WHAT_HIDE_TIP = 1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.PlayLineActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 1 || PlayLineActivity.this.moreAttachPopup2 == null) {
                return false;
            }
            PlayLineActivity.this.moreAttachPopup2.dismiss();
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.PlayLineActivity.12
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (SocketManager.RECEIVER_SOCKET_NOTIFY.equals(intent.getAction())) {
                int intExtra = intent.getIntExtra(SocketManager.EXTRA_SOCKET_ERROR, -1);
                String stringExtra = intent.getStringExtra(SocketManager.EXTRA_SOCKET_DATA);
                PlayLineActivity playLineActivity = PlayLineActivity.this;
                if (intExtra == 0) {
                    playLineActivity.mViewNetworkError.setVisibility(8);
                    PlayLineActivity.this.OnNotify(stringExtra);
                    return;
                }
                playLineActivity.mViewNetworkError.setVisibility(0);
                if (PlayLineActivity.this.loadPopupView != null && PlayLineActivity.this.loadPopupView.isShow()) {
                    PlayLineActivity.this.loadPopupView.delayDismiss(300L);
                }
                if ("onFailure".equals(stringExtra)) {
                    PlayLineActivity.this.onFailure();
                } else if ("onClosed".equals(stringExtra)) {
                    PlayLineActivity.this.onFailure();
                } else if ("NULL".equals(stringExtra)) {
                    PlayLineActivity.this.onClosed(stringExtra);
                }
            }
        }
    };

    private void hideLoading() {
        this.isLoading = false;
        this.mLayoutLoading.setVisibility(8);
        this.mLayoutLoading.setOnTouchListener(null);
    }

    private void init() {
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 2);
        this.mLayoutManager = gridLayoutManager;
        this.mRecyclerView.setLayoutManager(gridLayoutManager);
        this.mRoomList = new ArrayList();
        this.mUserList = new ArrayList();
        this.mWorldMessages = new ArrayList();
        LineRoomAdapter lineRoomAdapter = new LineRoomAdapter(this, this.mRoomList);
        this.mAdapter = lineRoomAdapter;
        this.mRecyclerView.setAdapter(lineRoomAdapter);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mWorldMessageLayoutManager = linearLayoutManager;
        this.mRecyclerWorldMessage.setLayoutManager(linearLayoutManager);
        LineRoomMessageAdapter lineRoomMessageAdapter = new LineRoomMessageAdapter(this, this.mWorldMessages);
        this.mWorldMessageAdapter = lineRoomMessageAdapter;
        this.mRecyclerWorldMessage.setAdapter(lineRoomMessageAdapter);
        SocketManager.getInstance().connSocket(2);
        showGuide();
        registerReceiver();
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleMore = (FrameLayout) findViewById(R.id.title_img_more);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        View findViewById = findViewById(R.id.view_network_error);
        this.mViewNetworkError = findViewById;
        this.mImgNetworkError = (ImageView) findViewById.findViewById(R.id.img_network_error);
        this.mTxtNetworkError = (TextView) this.mViewNetworkError.findViewById(R.id.txt_network_error);
        this.mTxtPeoples = (TextView) findViewById(R.id.play_line_txt_peoples);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.play_line_refreshLayout);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.play_line_recycler_view);
        this.mLayoutLoading = (LinearLayout) findViewById(R.id.play_line_layout_loading);
        this.mTvLoading = (TextView) findViewById(R.id.play_line_tv_loading);
        this.mImgPeopleBg = (ImageView) findViewById(R.id.img_peoples_bg);
        this.mRecyclerWorldMessage = (RecyclerView) findViewById(R.id.play_line_recycler_message);
        this.mEditMessage = (EditText) findViewById(R.id.play_line_edit_message);
        LanguageUtil.setText(this, this.mTitleTxtName, "gamesLobby");
        LanguageUtil.setText(this, this.mTvLoading, "isLoading");
        LanguageUtil.setText(this, this.mTxtNetworkError, "notNetwork");
    }

    private void registerNetReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction("android.net.conn.CONNECTIVITY_CHANGE");
        NetBroadcastReceiver netBroadcastReceiver = new NetBroadcastReceiver();
        this.mNetBroadcastReceiver = netBroadcastReceiver;
        netBroadcastReceiver.setNetChangeListener(this);
        AppUtil.registerReceiver(this, this.mNetBroadcastReceiver, intentFilter);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(SocketManager.RECEIVER_SOCKET_NOTIFY);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mTitleMore.setOnClickListener(this);
        this.mImgPeopleBg.setOnClickListener(this);
        this.mViewNetworkError.setOnClickListener(this);
        this.mAdapter.setOnItemClickListener(this);
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.PlayLineActivity.2
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                SocketManager.getInstance().worldUsers(false);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                if (!SocketManager.getInstance().isConnect()) {
                    SocketManager.getInstance().connSocket(2);
                } else {
                    SocketManager.getInstance().worldUsers(true);
                    SocketManager.getInstance().worldRooms(true);
                }
            }
        });
        this.mEditMessage.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.PlayLineActivity.3
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 4) {
                    return false;
                }
                String obj = textView.getText().toString();
                if (obj.length() > 128) {
                    return true;
                }
                textView.setText("");
                if (!TextUtils.isEmpty(obj)) {
                    SocketManager.getInstance().worldMessage(obj);
                }
                SoftKeyboardUtil.hideSoftKeyboard(PlayLineActivity.this, textView);
                return true;
            }
        });
    }

    private void showCreateDialog() {
        this.isPublic = 1;
        final Dialog dialog = new Dialog(this, R.style.MyDialog);
        View inflate = LayoutInflater.from(this).inflate(R.layout.dialog_line_create, (ViewGroup) null, false);
        LanguageUtil.setText(this, (TextView) inflate.findViewById(R.id.dialog_line_create_txt_title), "createRoom");
        LanguageUtil.setText(this, (TextView) inflate.findViewById(R.id.dialog_line_create_txt_room_name), "roomName");
        final EditText editText = (EditText) inflate.findViewById(R.id.dialog_line_create_edit_room_name);
        LanguageUtil.setHint(this, editText, "pleaseEnterRoomName");
        LanguageUtil.setText(this, (TextView) inflate.findViewById(R.id.dialog_line_create_txt_user_limit), "roomUserLimit");
        final TextView textView = (TextView) inflate.findViewById(R.id.txt_limit2);
        final TextView textView2 = (TextView) inflate.findViewById(R.id.txt_limit3);
        final TextView textView3 = (TextView) inflate.findViewById(R.id.txt_limit4);
        final TextView textView4 = (TextView) inflate.findViewById(R.id.txt_limit5);
        final String[] strArr = {"2", ExifInterface.GPS_MEASUREMENT_3D, "4", "5"};
        textView.setText("2");
        textView2.setText(strArr[1]);
        textView3.setText(strArr[2]);
        textView4.setText(strArr[3]);
        View.OnClickListener onClickListener = new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayLineActivity.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                int id = view.getId();
                if (id == R.id.txt_limit2) {
                    PlayLineActivity.this.limitIndex = 0;
                    textView.setBackgroundResource(R.drawable.bg_edit_style_18);
                    textView2.setBackground(null);
                    textView3.setBackground(null);
                    textView4.setBackground(null);
                    return;
                }
                if (id == R.id.txt_limit3) {
                    PlayLineActivity.this.limitIndex = 1;
                    textView.setBackground(null);
                    textView2.setBackgroundResource(R.drawable.bg_edit_style_18);
                    textView3.setBackground(null);
                    textView4.setBackground(null);
                    return;
                }
                if (id == R.id.txt_limit4) {
                    PlayLineActivity.this.limitIndex = 2;
                    textView.setBackground(null);
                    textView2.setBackground(null);
                    textView3.setBackgroundResource(R.drawable.bg_edit_style_18);
                    textView4.setBackground(null);
                    return;
                }
                if (id == R.id.txt_limit5) {
                    PlayLineActivity.this.limitIndex = 3;
                    textView.setBackground(null);
                    textView2.setBackground(null);
                    textView3.setBackground(null);
                    textView4.setBackgroundResource(R.drawable.bg_edit_style_18);
                }
            }
        };
        textView.setOnClickListener(onClickListener);
        textView2.setOnClickListener(onClickListener);
        textView3.setOnClickListener(onClickListener);
        textView4.setOnClickListener(onClickListener);
        LanguageUtil.setText(this, (TextView) inflate.findViewById(R.id.dialog_line_create_txt_room_type), "roomType");
        final TextView textView5 = (TextView) inflate.findViewById(R.id.dialog_line_create_txt_room_public);
        LanguageUtil.setText(this, textView5, "isPublic");
        final TextView textView6 = (TextView) inflate.findViewById(R.id.dialog_line_create_txt_room_private);
        LanguageUtil.setText(this, textView6, "isPrivate");
        Button button = (Button) inflate.findViewById(R.id.dialog_line_create_btn_complete);
        LanguageUtil.setText(this, button, "confirm");
        Button button2 = (Button) inflate.findViewById(R.id.dialog_line_create_btn_cancel);
        LanguageUtil.setText(this, button2, "cancel");
        button2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayLineActivity.7
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                Dialog dialog2 = dialog;
                if (dialog2 == null || !dialog2.isShowing()) {
                    return;
                }
                dialog.dismiss();
            }
        });
        textView5.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayLineActivity.8
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (PlayLineActivity.this.isPublic != 1) {
                    PlayLineActivity.this.isPublic = 1;
                    textView5.setBackground(ContextCompat.getDrawable(PlayLineActivity.this, R.drawable.bg_edit_style_18));
                    textView6.setBackground(null);
                }
            }
        });
        textView6.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayLineActivity.9
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (PlayLineActivity.this.isPublic != 0) {
                    PlayLineActivity.this.isPublic = 0;
                    textView5.setBackground(null);
                    textView6.setBackground(ContextCompat.getDrawable(PlayLineActivity.this, R.drawable.bg_edit_style_18));
                }
            }
        });
        button.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayLineActivity.10
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                String obj = editText.getText().toString();
                int intValue = TextUtils.isEmpty(strArr[PlayLineActivity.this.limitIndex]) ? 2 : Integer.valueOf(strArr[PlayLineActivity.this.limitIndex]).intValue();
                if (TextUtils.isEmpty(obj)) {
                    PlayLineActivity playLineActivity = PlayLineActivity.this;
                    ToastUtil.showShortToast(playLineActivity, LanguageUtil.getLanguageByKey(playLineActivity, "pleaseEnterRoomName"));
                } else if (SocketManager.getInstance().getWebSocket() != null) {
                    SocketManager.getInstance().createRoom(obj, intValue, PlayLineActivity.this.isPublic);
                    Dialog dialog2 = dialog;
                    if (dialog2 == null || !dialog2.isShowing()) {
                        return;
                    }
                    dialog.dismiss();
                }
            }
        });
        dialog.setCanceledOnTouchOutside(true);
        dialog.setContentView(inflate);
        Window window = dialog.getWindow();
        WindowManager.LayoutParams attributes = window.getAttributes();
        attributes.width = (int) (getWindowManager().getDefaultDisplay().getWidth() * 0.9d);
        window.setAttributes(attributes);
        dialog.show();
    }

    private void showGuide() {
        this.mTitleMore.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.PlayLineActivity.5
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                try {
                    if (MyApplication.getApplication().isLineCreateShowGuide()) {
                        String languageByKey = LanguageUtil.getLanguageByKey(PlayLineActivity.this, "createRoom");
                        PlayLineActivity.this.moreAttachPopup2 = new MoreAttachPopup2(PlayLineActivity.this, languageByKey);
                        PlayLineActivity.this.moreAttachPopup2.showDownPop(PlayLineActivity.this.mTitleMore);
                        MyApplication.getApplication().setLineCreateShowGuide(false);
                        PlayLineActivity.this.mHandler.sendEmptyMessageDelayed(1, 8000L);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void showLoading() {
        this.isLoading = true;
        this.mLayoutLoading.setVisibility(0);
        this.mLayoutLoading.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.PlayLineActivity.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                return true;
            }
        });
    }

    private void showLoadingConnLine() {
        SocketManager.getInstance().connSocket(2);
        BasePopupView basePopupView = this.loadPopupView;
        if (basePopupView == null || !basePopupView.isShow()) {
            this.loadPopupView = new XPopup.Builder(this).asLoading(LanguageUtil.getLanguageByKey(this, "isReconnecting")).show();
        }
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void OnNotify(String str) {
        List<SocketRoomModel> list;
        List<SocketUserModel> list2;
        try {
            JsonObject asJsonObject = new JsonParser().parse(str).getAsJsonObject();
            if (1 == asJsonObject.get("success").getAsInt()) {
                PlayLineSocketModel playLineSocketModel = (PlayLineSocketModel) this.gson.fromJson(str, PlayLineSocketModel.class);
                String str2 = "";
                String asString = asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION) == null ? "" : asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION).getAsString();
                if ("onlinePlay".equals(asString)) {
                    BasePopupView basePopupView = this.loadPopupView;
                    if (basePopupView != null && basePopupView.isShow()) {
                        this.loadPopupView.delayDismiss(300L);
                    }
                    this.mRefreshLayout.finishRefresh();
                    this.mRefreshLayout.finishLoadMore();
                    return;
                }
                if (!"createRoom".equals(asString) && !"roomInfo".equals(asString)) {
                    if ("enterRoom".equals(asString)) {
                        startActivity(new Intent(this, (Class<?>) LineRoomActivity.class));
                        return;
                    }
                    if (!"enterRoomWait".equals(asString) && !"enterRoomApply".equals(asString)) {
                        final int i = 0;
                        if ("inviteUserRoom".equals(asString)) {
                            if (asJsonObject.get("roomId") != null) {
                                i = asJsonObject.get("roomId").getAsInt();
                            }
                            if (asJsonObject.get("userName") != null) {
                                str2 = asJsonObject.get("userName").getAsString() + " " + LanguageUtil.getLanguageByKey(this, "inviteJoin");
                            }
                            if (i > 0) {
                                DialogUtil.showInviteDialog(this, str2, "cancel", "agreeButton", new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayLineActivity.13
                                    @Override // android.view.View.OnClickListener
                                    public void onClick(View view) {
                                        SocketManager.getInstance().enterRoom(i);
                                    }
                                });
                                return;
                            }
                            return;
                        }
                        if (!"worldUsers".equals(asString)) {
                            if ("worldRooms".equals(asString)) {
                                this.mRefreshLayout.finishRefresh();
                                this.mRefreshLayout.finishLoadMore();
                                if (playLineSocketModel.getHome() == 1 && (list = this.mRoomList) != null) {
                                    list.clear();
                                }
                                if (playLineSocketModel.getDelRoom() != null && playLineSocketModel.getDelRoom().length > 0) {
                                    for (int i2 = 0; i2 < playLineSocketModel.getDelRoom().length; i2++) {
                                        int i3 = 0;
                                        while (true) {
                                            if (i3 >= this.mRoomList.size()) {
                                                break;
                                            }
                                            if (playLineSocketModel.getDelRoom()[i2] == this.mRoomList.get(i3).getRoomId()) {
                                                this.mRoomList.remove(i3);
                                                break;
                                            }
                                            i3++;
                                        }
                                    }
                                }
                                if (playLineSocketModel.getRooms() != null && playLineSocketModel.getRooms().length > 0) {
                                    this.mRoomList.addAll(Arrays.asList(playLineSocketModel.getRooms()));
                                }
                                this.mAdapter.notifyDataSetChanged();
                                return;
                            }
                            return;
                        }
                        if (playLineSocketModel.getHome() == 1 && (list2 = this.mUserList) != null) {
                            list2.clear();
                        }
                        if (playLineSocketModel.getDelUser() != null && playLineSocketModel.getDelUser().length > 0) {
                            for (int i4 = 0; i4 < playLineSocketModel.getDelUser().length; i4++) {
                                int i5 = 0;
                                while (true) {
                                    if (i5 >= this.mUserList.size()) {
                                        break;
                                    }
                                    if (playLineSocketModel.getDelUser()[i4] == this.mUserList.get(i5).getUserId()) {
                                        this.mUserList.remove(i5);
                                        LineUserDialog lineUserDialog = this.mLineUserDialog;
                                        if (lineUserDialog != null) {
                                            lineUserDialog.removeUser(i5);
                                        }
                                    } else {
                                        i5++;
                                    }
                                }
                            }
                        }
                        if (playLineSocketModel.getUsers() != null && playLineSocketModel.getUsers().length > 0) {
                            List<SocketUserModel> asList = Arrays.asList(playLineSocketModel.getUsers());
                            this.mUserList.addAll(asList);
                            LineUserDialog lineUserDialog2 = this.mLineUserDialog;
                            if (lineUserDialog2 != null) {
                                lineUserDialog2.addUser(asList);
                            }
                        }
                        this.mTxtPeoples.setText(playLineSocketModel.getUserNum() + "");
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.receiver.NetBroadcastReceiver.NetChangeListener
    public void onChangeListener(int i) {
        if (i == 2) {
            LogUtil.e("SocketManager 网络状态：WIFI");
            int i2 = this.mLastNetStatus;
            if (i2 != -1 && i2 != 2) {
                showLoadingConnLine();
            }
            this.mLastNetStatus = 2;
            return;
        }
        if (i != 1) {
            if (i == 0) {
                LogUtil.e("SocketManager 网络状态：没网络");
                this.mLastNetStatus = 0;
                return;
            }
            return;
        }
        LogUtil.e("SocketManager 网络状态：移动网络");
        int i3 = this.mLastNetStatus;
        if (i3 != -1 && i3 != 1) {
            showLoadingConnLine();
        }
        this.mLastNetStatus = 1;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.title_img_more) {
            showCreateDialog();
            return;
        }
        if (id == R.id.img_peoples_bg) {
            LineUserDialog lineUserDialog = new LineUserDialog(this, this.mUserList, "lobbyUserList", 1);
            this.mLineUserDialog = lineUserDialog;
            lineUserDialog.show();
        } else if (id == R.id.view_network_error) {
            showLoadingConnLine();
        }
    }

    public void onClosed(String str) {
        if ("NULL".equals(str)) {
            SocketManager.getInstance().connSocket(2);
        }
        try {
            SmartRefreshLayout smartRefreshLayout = this.mRefreshLayout;
            if (smartRefreshLayout != null) {
                smartRefreshLayout.finishRefresh();
                this.mRefreshLayout.finishLoadMore();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_play_line);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        SocketManager.getInstance().clearHandler();
        SocketManager.getInstance().close();
        SocketManager.getInstance().closeSocket();
        unRegisterReceiver();
    }

    public void onFailure() {
        try {
            SmartRefreshLayout smartRefreshLayout = this.mRefreshLayout;
            if (smartRefreshLayout != null) {
                smartRefreshLayout.finishRefresh();
                this.mRefreshLayout.finishLoadMore();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.adapter.LineRoomAdapter.OnItemClickListener
    public void onItemClick(final int i) {
        DialogUtil.showConfirmDialog(this, "joinRoom", LanguageUtil.getLanguageByKey(this, "roomName") + this.mRoomList.get(i).getRoomName(), null, this.mRoomList.get(i).getIsPublic() == 1 ? LanguageUtil.getLanguageByKey(this, "joinButton") : LanguageUtil.getLanguageByKey(this, "applicationButton"), new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayLineActivity.11
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (PlayLineActivity.this.mRoomList.get(i) != null) {
                    SocketManager.getInstance().enterRoom(((SocketRoomModel) PlayLineActivity.this.mRoomList.get(i)).getRoomId());
                }
                DialogUtil.dismissDialog();
            }
        });
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onPause() {
        super.onPause();
        LogUtil.e("onPause");
        try {
            if (this.mEditMessage.hasFocus()) {
                this.mEditMessage.clearFocus();
            }
            NetBroadcastReceiver netBroadcastReceiver = this.mNetBroadcastReceiver;
            if (netBroadcastReceiver != null) {
                unregisterReceiver(netBroadcastReceiver);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onResume() {
        super.onResume();
        LogUtil.e("onResume");
        registerNetReceiver();
    }
}
