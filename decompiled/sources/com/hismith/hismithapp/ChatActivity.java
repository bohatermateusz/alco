package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.Editable;
import android.text.TextUtils;
import android.util.Pair;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.facebook.internal.AnalyticsEvents;
import com.facebook.internal.NativeProtocol;
import com.facebook.share.internal.ShareConstants;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.adapter.ChatAdapter;
import com.hismith.hismithapp.adapter.ChatFunAdapter;
import com.hismith.hismithapp.adapter.EmojiAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.db.DBMessageUtil;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.dialog.MoreDialog;
import com.hismith.hismithapp.dialog.ReportSelectDialog;
import com.hismith.hismithapp.fragment.PersonalMessageFragment;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.PictureResultManager;
import com.hismith.hismithapp.model.BlockModel;
import com.hismith.hismithapp.model.ChatFunModel;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.MessageModel.Message;
import com.hismith.hismithapp.model.MsgSendModel;
import com.hismith.hismithapp.model.ReferenceMsg;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.picker.PictureSelectorManager;
import com.hismith.hismithapp.service.HuDongPlayService;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.FileUtil;
import com.hismith.hismithapp.utils.ImageCompressor;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.RecorderUtil;
import com.hismith.hismithapp.utils.RequestBodyUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.luck.picture.lib.config.PictureMimeType;
import com.luck.picture.lib.entity.LocalMedia;
import com.lxj.xpopup.util.KeyboardUtils;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshListener;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.Observable;
import io.reactivex.ObservableSource;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import io.reactivex.functions.Function;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TimeZone;
import okhttp3.RequestBody;

/* loaded from: classes3.dex */
public class ChatActivity extends BaseActivity implements View.OnClickListener {
    private static final int RECORDER_CANCEL = 2;
    private static final int RECORDER_START = 0;
    private static final int RECORDER_STOP = 1;
    public static final int RESULT_ALIAS = 1;
    private RetrofitRequest baseRequest;
    private ClipboardManager clipboard;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbRead;
    private SQLiteDatabase dbWrite;
    private RetrofitRequest fileRequest;
    private ChatAdapter mAdapter;
    private EditText mEditMessage;
    private EmojiAdapter mEmojiAdapter;
    private GridLayoutManager mEmojiLayoutManager;
    private ChatFunAdapter mFunAdapter;
    private GridLayoutManager mFunLayoutManager;
    private List<ChatFunModel> mFunList;
    private ImageView mImgEmoji;
    private ImageView mImgFun;
    private ImageView mImgVoice;
    private ImageView mImgVoiceStatus;
    private LinearLayout mLayoutEmoji;
    private LinearLayout mLayoutFun;
    private LinearLayoutManager mLayoutManager;
    private LinearLayout mLayoutVoiceStatus;
    private List<MessageModel.Message> mList;
    private MessageModel.Message mMessageModel;
    private RecyclerView mRecyclerEmoji;
    private RecyclerView mRecyclerFun;
    private RecyclerView mRecyclerView;
    private ImageView mReferenceClose;
    private TextView mReferenceContent;
    private ConstraintLayout mReferenceLayout;
    private SmartRefreshLayout mRefreshLayout;
    private ImageView mTitleImgBack;
    private ImageView mTitleImgMore;
    private TextView mTitleTxtName;
    private TextView mTxtVoice;
    private TextView mTxtVoiceStatus;
    private UserModel mUserModel;
    private ActivityResultLauncher<Intent> nameResultLauncher;
    private PictureResultManager pictureResultManager;
    private RecorderUtil recorderUtil;
    private int firstChat = 0;
    private int lastMsgId = 0;
    private int firstMsgId = 0;
    private int pageCount = 10;
    private boolean isSendText = true;
    private float downY = 0.0f;
    private float moveY = 0.0f;
    private boolean isCancelVoice = false;
    private int voiceLength = 0;
    private int action = -1;
    private MessageModel.Message mReferenceMessage = null;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private final int REQUEST_CAMERA = 777;
    private final int REQUEST_ALBUM = 888;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.ChatActivity.2
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 0) {
                ChatActivity.this.recorderUtil.startRecording("VOICE_" + String.valueOf(System.currentTimeMillis()) + PictureMimeType.MP4);
            } else if (i == 1) {
                ChatActivity.this.recorderUtil.stopRecording();
                if (ChatActivity.this.recorderUtil.getTimeInterval() >= 1000) {
                    try {
                        ChatActivity chatActivity = ChatActivity.this;
                        chatActivity.voiceLength = (int) chatActivity.recorderUtil.getTimeInterval();
                        ChatActivity chatActivity2 = ChatActivity.this;
                        chatActivity2.sendFileMessage(ShareConstants.WEB_DIALOG_PARAM_MEDIA, chatActivity2.recorderUtil.getFilePath());
                        ChatActivity.this.voiceLength = 0;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            } else if (i == 2) {
                ChatActivity.this.recorderUtil.cancelRecording();
            }
            return false;
        }
    });
    private View.OnTouchListener onVoiceTouchListener = new View.OnTouchListener() { // from class: com.hismith.hismithapp.ChatActivity.23
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                LogUtil.e("onVoiceTouchListener, action:+MotionEvent.ACTION_DOWN");
                ChatActivity.this.action = 0;
                ChatActivity.this.downY = motionEvent.getY();
                ChatActivity.this.initRecordPermission();
            } else if (action == 1) {
                LogUtil.e("onVoiceTouchListener, action:+MotionEvent.ACTION_UP");
                ChatActivity.this.action = 1;
                ChatActivity.this.mLayoutVoiceStatus.setVisibility(8);
                float y = motionEvent.getY();
                ChatActivity.this.recorderUtil.stopRecording();
                LogUtil.e("onVoiceTouchListener, action:+MotionEvent.ACTION_UP (downY - upY)：" + (ChatActivity.this.downY - y) + ", TimeInterval:" + ChatActivity.this.recorderUtil.getTimeInterval());
                if (ChatActivity.this.downY - y > 150.0f || ChatActivity.this.recorderUtil.getTimeInterval() < 1) {
                    ChatActivity.this.recorderUtil.cancelRecording();
                } else {
                    try {
                        try {
                            ChatActivity chatActivity = ChatActivity.this;
                            chatActivity.voiceLength = (int) chatActivity.recorderUtil.getTimeInterval();
                            ChatActivity chatActivity2 = ChatActivity.this;
                            chatActivity2.sendFileMessage("voice", chatActivity2.recorderUtil.getFilePath());
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    } finally {
                        ChatActivity.this.voiceLength = 0;
                    }
                }
            } else if (action == 2) {
                ChatActivity.this.moveY = motionEvent.getY();
                float f = ChatActivity.this.downY - ChatActivity.this.moveY;
                LogUtil.e("onVoiceTouchListener, action:+MotionEvent.ACTION_MOVE ty：" + f);
                if (f > 150.0f && !ChatActivity.this.isCancelVoice) {
                    ChatActivity.this.isCancelVoice = true;
                    ChatActivity.this.mLayoutVoiceStatus.setBackground(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.bg_chat_voice_cancel));
                    ChatActivity.this.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.ic_chat_voice_cancel));
                    ChatActivity chatActivity3 = ChatActivity.this;
                    LanguageUtil.setText(chatActivity3, chatActivity3.mTxtVoiceStatus, "releaseYourFingerUpCancel");
                } else if (f <= 150.0f && ChatActivity.this.isCancelVoice) {
                    ChatActivity.this.isCancelVoice = false;
                    ChatActivity.this.mLayoutVoiceStatus.setBackground(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.bg_chat_voice));
                    ChatActivity.this.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.ic_chat_voice_volume_1));
                    ChatActivity chatActivity4 = ChatActivity.this;
                    LanguageUtil.setText(chatActivity4, chatActivity4.mTxtVoiceStatus, "moveYourFingerUpCancel");
                }
            } else if (action == 3) {
                LogUtil.e("onVoiceTouchListener, action:+MotionEvent.ACTION_CANCEL");
            }
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.ChatActivity.24
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (!action.equals(Constans.RECEIVER_MESSAGE_NEW)) {
                if (SocketManager.RECEIVER_SOCKET_NOTIFY.equals(action)) {
                    int intExtra = intent.getIntExtra(SocketManager.EXTRA_SOCKET_ERROR, -1);
                    String stringExtra = intent.getStringExtra(SocketManager.EXTRA_SOCKET_DATA);
                    LogUtil.e("action:" + action + ", data:" + stringExtra);
                    ChatActivity.this.initSocketData(intExtra, stringExtra);
                    return;
                }
                return;
            }
            String stringExtra2 = intent.getStringExtra("msgId");
            String stringExtra3 = intent.getStringExtra("revokeId");
            String stringExtra4 = intent.getStringExtra("rejectId");
            int intExtra2 = intent.getIntExtra("newMessage", 0);
            LogUtil.e("action:" + action + ", msgId:" + stringExtra2 + ", newMessage:" + intExtra2);
            intent.getIntExtra("newMessageTotal", 0);
            intent.getIntExtra("newFirstChat", 0);
            intent.getIntExtra("newInteractive", 0);
            intent.getIntExtra("newNotice", 0);
            if (!TextUtils.isEmpty(stringExtra3)) {
                String[] split = stringExtra3.split(",");
                for (int i = 0; i < ChatActivity.this.mList.size(); i++) {
                    for (int i2 = 0; i2 < split.length; i2++) {
                        try {
                            if (!TextUtils.isEmpty(split[i2]) && ((MessageModel.Message) ChatActivity.this.mList.get(i)).getMsgId() == Integer.parseInt(split[i2])) {
                                ChatActivity.this.getNewDBMessageForMsgId(((MessageModel.Message) ChatActivity.this.mList.get(i)).getMsgId() + "");
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
            if (!TextUtils.isEmpty(stringExtra4)) {
                String[] split2 = stringExtra4.split(",");
                for (int i3 = 0; i3 < ChatActivity.this.mList.size(); i3++) {
                    for (int i4 = 0; i4 < split2.length; i4++) {
                        try {
                            if (!TextUtils.isEmpty(split2[i4]) && ((MessageModel.Message) ChatActivity.this.mList.get(i3)).getMsgId() == Integer.parseInt(split2[i4])) {
                                ChatActivity.this.getNewDBMessageForMsgId(((MessageModel.Message) ChatActivity.this.mList.get(i3)).getMsgId() + "");
                            }
                        } catch (Exception e2) {
                            e2.printStackTrace();
                        }
                    }
                }
            }
            if (!TextUtils.isEmpty(stringExtra2)) {
                ChatActivity.this.getNewDBMessageForMsgId(stringExtra2);
            }
            if (intExtra2 > 0) {
                ChatActivity.this.getNewDBMessageForUser();
            }
        }
    };

    private Map<String, RequestBody> buildUploadParams(String str, String str2) {
        String str3 = (String) SharedPreferencesUtil.get(this, Constans.Token);
        HashMap hashMap = new HashMap();
        RequestBodyUtil.addMultiPart(this, hashMap, Constans.Token, str3);
        RequestBodyUtil.addMultiPart(this, hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
        RequestBodyUtil.addMultiPart(this, hashMap, "fileType", str);
        if (!PictureMimeType.isContent(str2)) {
            File file = new File(str2);
            RequestBodyUtil.addMultiPart(this, hashMap, "requestId", file.getName());
            RequestBodyUtil.addMultiPart(this, hashMap, "file", file);
            return hashMap;
        }
        Uri parse = Uri.parse(str2);
        String fileNameFromUri = RequestBodyUtil.getFileNameFromUri(this, parse);
        RequestBodyUtil.addMultiPart(this, hashMap, "requestId", fileNameFromUri);
        RequestBodyUtil.addMultiPartUri(this, hashMap, fileNameFromUri, parse);
        return hashMap;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void clearChatHistoryDB() {
        if (this.mUserModel == null) {
            return;
        }
        try {
            this.dbWrite.delete(DatabaseHelper.TABLE_MESSAGE, "userId = ? and sendUserId = ?", new String[]{this.mUserModel.getUserId() + "", MyApplication.getApplication().getmUserModel().getUserId() + ""});
            this.mList.clear();
            this.mAdapter.notifyDataSetChanged();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void clearReference() {
        if (this.mReferenceLayout.getVisibility() != 8) {
            this.mReferenceLayout.setVisibility(8);
        }
        this.mReferenceMessage = null;
    }

    private void closeDB() {
        SQLiteDatabase sQLiteDatabase = this.dbRead;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
        }
        SQLiteDatabase sQLiteDatabase2 = this.dbWrite;
        if (sQLiteDatabase2 != null) {
            sQLiteDatabase2.close();
        }
        DatabaseHelper databaseHelper = this.dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(String str, String str2) {
        this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.mUserModel.getUserId()), "user", str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.ChatActivity.14
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(ChatActivity.this, LanguageUtil.getLanguageByKey(ChatActivity.this, "reportedSuccess"));
                }
            }
        });
    }

    private void compressImage(String str, String str2, File file) {
        LogUtil.e("文件压缩后大小：" + FileUtil.getFileSize(file));
        sendFileMessage("image", file.getPath());
    }

    private void getChatUpData() {
        if (this.mMessageModel != null) {
            this.mList.clear();
            this.mList.add(this.mMessageModel);
            this.mAdapter.notifyDataSetChanged();
        }
    }

    private boolean getDBMessageForUser() {
        Cursor query;
        String str = "msgId";
        if (this.mUserModel == null) {
            return false;
        }
        try {
            int i = 0;
            Cursor query2 = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE, null, "userId = ? and sendUserId = ?", new String[]{this.mUserModel.getUserId() + "", MyApplication.getApplication().getmUserModel().getUserId() + ""}, null, null, "msgId DESC", "0," + this.pageCount);
            if (query2 == null || !query2.moveToLast()) {
                return false;
            }
            this.firstMsgId = query2.getInt(query2.getColumnIndexOrThrow("msgId"));
            while (true) {
                MessageModel messageModel = new MessageModel();
                Objects.requireNonNull(messageModel);
                MessageModel.Message message = messageModel.new Message();
                int i2 = query2.getInt(query2.getColumnIndexOrThrow(str));
                this.lastMsgId = i2;
                message.setMsgId(i2);
                int i3 = query2.getInt(query2.getColumnIndexOrThrow("userId"));
                message.setUserId(i3);
                message.setVoiceLength(query2.getInt(query2.getColumnIndexOrThrow("voiceLength")));
                message.setType(query2.getString(query2.getColumnIndexOrThrow("type")));
                message.setRevoke(query2.getInt(query2.getColumnIndexOrThrow("revoke")));
                message.setReject(query2.getInt(query2.getColumnIndexOrThrow("reject")));
                message.setDate(query2.getString(query2.getColumnIndexOrThrow("date")));
                message.setContent(query2.getString(query2.getColumnIndexOrThrow(FirebaseAnalytics.Param.CONTENT)));
                int columnIndexOrThrow = query2.getColumnIndexOrThrow("localPath");
                if (columnIndexOrThrow != -1) {
                    message.setLocalPath(query2.getString(columnIndexOrThrow));
                }
                message.setAndroidRead(query2.getInt(query2.getColumnIndexOrThrow("androidRead")));
                message.setAndroidSend(query2.getInt(query2.getColumnIndexOrThrow("androidSend")));
                message.setAlias(query2.getString(query2.getColumnIndexOrThrow("alias")));
                message.setStatus(i);
                message.setReferenceId(query2.getInt(query2.getColumnIndexOrThrow("referenceId")));
                String str2 = str;
                Cursor query3 = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE_USER, null, "userId = ?", new String[]{i3 + ""}, null, null, null);
                if (query3 != null && query3.moveToFirst()) {
                    message.setName(query3.getString(query3.getColumnIndexOrThrow("name")));
                    message.setKing(query3.getInt(query3.getColumnIndexOrThrow("king")));
                    if (query3.getColumnIndexOrThrow("gradeIcon") != -1) {
                        message.setGradeIcon(query3.getString(query3.getColumnIndexOrThrow("gradeIcon")));
                    }
                    message.setHeadIcon(query3.getString(query3.getColumnIndexOrThrow("headIcon")));
                    message.setGender(query3.getInt(query3.getColumnIndexOrThrow("gender")));
                    message.setConstellation(query3.getString(query3.getColumnIndexOrThrow("constellation")));
                    message.setAlias(query3.getString(query3.getColumnIndexOrThrow("alias")));
                    query3.close();
                }
                if (message.getReferenceId() > 0 && (query = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE_REFERENCE, null, "referenceId = ? and msgId = ?", new String[]{message.getReferenceId() + "", message.getMsgId() + ""}, null, null, null)) != null && query.moveToFirst()) {
                    ReferenceMsg referenceMsg = new ReferenceMsg();
                    referenceMsg.setReferenceId(query.getInt(query.getColumnIndexOrThrow("referenceId")));
                    referenceMsg.setType(query.getString(query.getColumnIndexOrThrow("type")));
                    referenceMsg.setContent(query.getString(query.getColumnIndexOrThrow(FirebaseAnalytics.Param.CONTENT)));
                    referenceMsg.setVoiceLength(query.getInt(query.getColumnIndexOrThrow("voiceLength")));
                    referenceMsg.setUserId(query.getInt(query.getColumnIndexOrThrow("userId")));
                    referenceMsg.setUserName(query.getString(query.getColumnIndexOrThrow("userName")));
                    int columnIndexOrThrow2 = query.getColumnIndexOrThrow("localPath");
                    if (columnIndexOrThrow2 != -1) {
                        referenceMsg.setLocalPath(query.getString(columnIndexOrThrow2));
                    }
                    message.setReferenceMsg(referenceMsg);
                    query.close();
                }
                this.mList.add(message);
                if (!query2.moveToPrevious()) {
                    this.mAdapter.notifyDataSetChanged();
                    this.mLayoutManager.scrollToPositionWithOffset(this.mList.size() - 1, Integer.MIN_VALUE);
                    query2.close();
                    updateDBManagerRead();
                    return true;
                }
                str = str2;
                i = 0;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getNewDBMessageForMsgId(String str) {
        try {
            LogUtil.e("根据MsgId更新信息");
            Cursor query = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE, null, "msgId = ?", new String[]{str}, null, null, null);
            if (query != null && query.moveToFirst()) {
                for (int i = 0; i < this.mList.size(); i++) {
                    if (this.mList.get(i).getReferenceId() == Integer.parseInt(str)) {
                        this.mList.get(i).setReferenceMsg(null);
                        this.mAdapter.notifyItemChanged(i);
                    }
                    if (str.equals(this.mList.get(i).getMsgId() + "")) {
                        this.mList.get(i).setRevoke(query.getInt(query.getColumnIndexOrThrow("revoke")));
                        this.mList.get(i).setReject(query.getInt(query.getColumnIndexOrThrow("reject")));
                        this.mList.get(i).setVoiceLength(query.getInt(query.getColumnIndexOrThrow("voiceLength")));
                        this.mAdapter.notifyItemChanged(i, "");
                    }
                }
            }
            if (query != null) {
                query.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getNewDBMessageForUser() {
        if (this.mUserModel == null) {
            return;
        }
        try {
            int i = 0;
            Cursor query = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE, null, "userId = ? and msgId > ? and sendUserId = ?", new String[]{this.mUserModel.getUserId() + "", this.lastMsgId + "", MyApplication.getApplication().getmUserModel().getUserId() + ""}, null, null, "msgId DESC");
            if (query != null && query.moveToLast()) {
                while (true) {
                    MessageModel messageModel = new MessageModel();
                    Objects.requireNonNull(messageModel);
                    MessageModel.Message message = messageModel.new Message();
                    int i2 = query.getInt(query.getColumnIndexOrThrow("msgId"));
                    this.lastMsgId = i2;
                    message.setMsgId(i2);
                    int i3 = query.getInt(query.getColumnIndexOrThrow("userId"));
                    message.setUserId(i3);
                    message.setVoiceLength(query.getInt(query.getColumnIndexOrThrow("voiceLength")));
                    message.setType(query.getString(query.getColumnIndexOrThrow("type")));
                    message.setRevoke(query.getInt(query.getColumnIndexOrThrow("revoke")));
                    message.setReject(query.getInt(query.getColumnIndexOrThrow("reject")));
                    message.setDate(query.getString(query.getColumnIndexOrThrow("date")));
                    message.setContent(query.getString(query.getColumnIndexOrThrow(FirebaseAnalytics.Param.CONTENT)));
                    message.setLocalPath(query.getString(query.getColumnIndexOrThrow("localPath")));
                    message.setAndroidRead(query.getInt(query.getColumnIndexOrThrow("androidRead")));
                    message.setAndroidSend(query.getInt(query.getColumnIndexOrThrow("androidSend")));
                    message.setAlias(query.getString(query.getColumnIndexOrThrow("alias")));
                    message.setStatus(i);
                    message.setReferenceId(query.getInt(query.getColumnIndexOrThrow("referenceId")));
                    SQLiteDatabase sQLiteDatabase = this.dbRead;
                    String str = DatabaseHelper.TABLE_MESSAGE_USER;
                    int i4 = i;
                    String[] strArr = new String[1];
                    strArr[i4] = i3 + "";
                    Cursor query2 = sQLiteDatabase.query(str, null, "userId = ?", strArr, null, null, null);
                    if (query2 != null && query2.moveToFirst()) {
                        message.setName(query2.getString(query2.getColumnIndexOrThrow("name")));
                        message.setKing(query2.getInt(query2.getColumnIndexOrThrow("king")));
                        if (query2.getColumnIndexOrThrow("gradeIcon") != -1) {
                            message.setGradeIcon(query2.getString(query2.getColumnIndexOrThrow("gradeIcon")));
                        }
                        message.setHeadIcon(query2.getString(query2.getColumnIndexOrThrow("headIcon")));
                        message.setGender(query2.getInt(query2.getColumnIndexOrThrow("gender")));
                        message.setConstellation(query2.getString(query2.getColumnIndexOrThrow("constellation")));
                        message.setAlias(query2.getString(query2.getColumnIndexOrThrow("alias")));
                        query2.close();
                    }
                    if (message.getReferenceId() > 0) {
                        SQLiteDatabase sQLiteDatabase2 = this.dbRead;
                        String str2 = DatabaseHelper.TABLE_MESSAGE_REFERENCE;
                        String[] strArr2 = new String[2];
                        strArr2[i4] = message.getReferenceId() + "";
                        strArr2[1] = message.getMsgId() + "";
                        Cursor query3 = sQLiteDatabase2.query(str2, null, "referenceId = ? and msgId = ?", strArr2, null, null, null);
                        if (query3 != null && query3.moveToFirst()) {
                            ReferenceMsg referenceMsg = new ReferenceMsg();
                            referenceMsg.setReferenceId(query3.getInt(query3.getColumnIndexOrThrow("referenceId")));
                            referenceMsg.setType(query3.getString(query3.getColumnIndexOrThrow("type")));
                            referenceMsg.setContent(query3.getString(query3.getColumnIndexOrThrow(FirebaseAnalytics.Param.CONTENT)));
                            referenceMsg.setVoiceLength(query3.getInt(query3.getColumnIndexOrThrow("voiceLength")));
                            referenceMsg.setUserId(query3.getInt(query3.getColumnIndexOrThrow("userId")));
                            referenceMsg.setUserName(query3.getString(query3.getColumnIndexOrThrow("userName")));
                            int columnIndexOrThrow = query3.getColumnIndexOrThrow("localPath");
                            if (columnIndexOrThrow != -1) {
                                referenceMsg.setLocalPath(query3.getString(columnIndexOrThrow));
                            }
                            message.setReferenceMsg(referenceMsg);
                            query3.close();
                        }
                    }
                    this.mList.add(message);
                    if (!query.moveToPrevious()) {
                        break;
                    } else {
                        i = i4;
                    }
                }
            }
            this.mAdapter.notifyDataSetChanged();
            this.mRecyclerView.smoothScrollToPosition(this.mList.size() - 1);
            if (query != null) {
                query.close();
            }
            updateDBManagerRead();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    /* JADX WARN: Multi-variable type inference failed */
    public boolean getOldDBMessageForUser() {
        boolean z;
        int i;
        int i2;
        String str;
        String str2 = "gradeIcon";
        if (this.mUserModel == null) {
            return false;
        }
        try {
            int i3 = 0;
            int i4 = 1;
            Cursor query = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE, null, "userId = ? and msgId < ? and sendUserId = ?", new String[]{this.mUserModel.getUserId() + "", this.firstMsgId + "", MyApplication.getApplication().getmUserModel().getUserId() + ""}, null, null, "msgId DESC", "0," + this.pageCount);
            if (query == null || !query.moveToFirst()) {
                z = 1;
                i = 0;
            } else {
                i = 0;
                while (true) {
                    i += i4;
                    MessageModel messageModel = new MessageModel();
                    Objects.requireNonNull(messageModel);
                    MessageModel.Message message = messageModel.new Message();
                    int i5 = query.getInt(query.getColumnIndexOrThrow("msgId"));
                    this.firstMsgId = i5;
                    message.setMsgId(i5);
                    int i6 = query.getInt(query.getColumnIndexOrThrow("userId"));
                    message.setUserId(i6);
                    message.setVoiceLength(query.getInt(query.getColumnIndexOrThrow("voiceLength")));
                    message.setType(query.getString(query.getColumnIndexOrThrow("type")));
                    message.setRevoke(query.getInt(query.getColumnIndexOrThrow("revoke")));
                    message.setReject(query.getInt(query.getColumnIndexOrThrow("reject")));
                    message.setDate(query.getString(query.getColumnIndexOrThrow("date")));
                    message.setContent(query.getString(query.getColumnIndexOrThrow(FirebaseAnalytics.Param.CONTENT)));
                    message.setLocalPath(query.getString(query.getColumnIndexOrThrow("localPath")));
                    message.setAndroidRead(query.getInt(query.getColumnIndexOrThrow("androidRead")));
                    message.setAndroidSend(query.getInt(query.getColumnIndexOrThrow("androidSend")));
                    message.setAlias(query.getString(query.getColumnIndexOrThrow("alias")));
                    message.setStatus(i3);
                    message.setReferenceId(query.getInt(query.getColumnIndexOrThrow("referenceId")));
                    SQLiteDatabase sQLiteDatabase = this.dbRead;
                    String str3 = DatabaseHelper.TABLE_MESSAGE_USER;
                    String[] strArr = new String[i4];
                    i2 = i4;
                    strArr[0] = i6 + "";
                    Cursor query2 = sQLiteDatabase.query(str3, null, "userId = ?", strArr, null, null, null);
                    if (query2 != null && query2.moveToFirst()) {
                        message.setName(query2.getString(query2.getColumnIndexOrThrow("name")));
                        message.setKing(query2.getInt(query2.getColumnIndexOrThrow("king")));
                        if (query2.getColumnIndexOrThrow(str2) != -1) {
                            message.setGradeIcon(query2.getString(query2.getColumnIndexOrThrow(str2)));
                        }
                        message.setHeadIcon(query2.getString(query2.getColumnIndexOrThrow("headIcon")));
                        message.setGender(query2.getInt(query2.getColumnIndexOrThrow("gender")));
                        message.setConstellation(query2.getString(query2.getColumnIndexOrThrow("constellation")));
                        message.setAlias(query2.getString(query2.getColumnIndexOrThrow("alias")));
                        query2.close();
                    }
                    if (message.getReferenceId() > 0) {
                        SQLiteDatabase sQLiteDatabase2 = this.dbRead;
                        String str4 = DatabaseHelper.TABLE_MESSAGE_REFERENCE;
                        String[] strArr2 = new String[2];
                        str = str2;
                        strArr2[0] = message.getReferenceId() + "";
                        strArr2[i2 == true ? 1 : 0] = message.getMsgId() + "";
                        Cursor query3 = sQLiteDatabase2.query(str4, null, "referenceId = ? and msgId = ?", strArr2, null, null, null);
                        if (query3 != null && query3.moveToFirst()) {
                            ReferenceMsg referenceMsg = new ReferenceMsg();
                            referenceMsg.setReferenceId(query3.getInt(query3.getColumnIndexOrThrow("referenceId")));
                            referenceMsg.setType(query3.getString(query3.getColumnIndexOrThrow("type")));
                            referenceMsg.setContent(query3.getString(query3.getColumnIndexOrThrow(FirebaseAnalytics.Param.CONTENT)));
                            referenceMsg.setVoiceLength(query3.getInt(query3.getColumnIndexOrThrow("voiceLength")));
                            referenceMsg.setUserId(query3.getInt(query3.getColumnIndexOrThrow("userId")));
                            referenceMsg.setUserName(query3.getString(query3.getColumnIndexOrThrow("userName")));
                            int columnIndexOrThrow = query3.getColumnIndexOrThrow("localPath");
                            if (columnIndexOrThrow != -1) {
                                referenceMsg.setLocalPath(query3.getString(columnIndexOrThrow));
                            }
                            message.setReferenceMsg(referenceMsg);
                            query3.close();
                        }
                    } else {
                        str = str2;
                    }
                    this.mList.add(0, message);
                    if (!query.moveToNext()) {
                        break;
                    }
                    i4 = i2 == true ? 1 : 0;
                    str2 = str;
                    i3 = 0;
                }
                z = i2;
            }
            this.mAdapter.notifyDataSetChanged();
            this.mLayoutManager.scrollToPositionWithOffset(i, 0);
            if (query != null) {
                query.close();
            }
            return z;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private void getUserInfo() {
        if (this.mUserModel == null) {
            return;
        }
        this.baseRequest.getApi().getUserInfo((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.mUserModel.getUserId())).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.ChatActivity.13
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel.getSuccess() == 1) {
                    ChatActivity.this.mUserModel = userModel;
                }
            }
        });
    }

    private void init() {
        if (getIntent() != null) {
            this.mUserModel = (UserModel) getIntent().getSerializableExtra(Constans.UserModel);
            this.mMessageModel = (MessageModel.Message) getIntent().getSerializableExtra("messageModel");
            this.firstChat = getIntent().getIntExtra("firstChat", 0);
            getIntent().getBooleanExtra("refreshUser", false);
        }
        UserModel userModel = this.mUserModel;
        if (userModel != null) {
            this.mTitleTxtName.setText(userModel.getAlias());
        }
        this.baseRequest = new RetrofitRequest();
        this.fileRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_FILE);
        this.pictureResultManager = new PictureResultManager(this);
        this.clipboard = (ClipboardManager) getSystemService("clipboard");
        DatabaseHelper databaseHelper = new DatabaseHelper(this);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        this.dbRead = this.dbHepler.getReadableDatabase();
        this.recorderUtil = new RecorderUtil();
        this.mRefreshLayout.setEnableLoadMore(false);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        linearLayoutManager.setStackFromEnd(true);
        this.mRecyclerView.setLayoutManager(this.mLayoutManager);
        this.mList = new ArrayList();
        ChatAdapter chatAdapter = new ChatAdapter(this, this.mLayoutManager, this.mList);
        this.mAdapter = chatAdapter;
        this.mRecyclerView.setAdapter(chatAdapter);
        initEmoji();
        initFun();
        if (this.firstChat == 0) {
            getDBMessageForUser();
        } else {
            getChatUpData();
        }
        registerActivityResult();
        registerReceiver();
    }

    private void initEmoji() {
        this.mEmojiLayoutManager = new GridLayoutManager(this, 7);
        EmojiAdapter emojiAdapter = new EmojiAdapter(this);
        this.mEmojiAdapter = emojiAdapter;
        emojiAdapter.setOnItemCLickListener(new EmojiAdapter.OnItemCLickListener() { // from class: com.hismith.hismithapp.ChatActivity.11
            @Override // com.hismith.hismithapp.adapter.EmojiAdapter.OnItemCLickListener
            public void onItemClick(String str) {
                int selectionStart = ChatActivity.this.mEditMessage.getSelectionStart();
                Editable editableText = ChatActivity.this.mEditMessage.getEditableText();
                if (selectionStart < 0 || selectionStart >= editableText.length()) {
                    editableText.append((CharSequence) str);
                } else {
                    editableText.insert(selectionStart, str);
                }
            }
        });
        this.mRecyclerEmoji.setLayoutManager(this.mEmojiLayoutManager);
        this.mRecyclerEmoji.setAdapter(this.mEmojiAdapter);
    }

    private void initFun() {
        this.mFunLayoutManager = new GridLayoutManager(this, 4);
        ArrayList arrayList = new ArrayList();
        this.mFunList = arrayList;
        arrayList.add(new ChatFunModel(LanguageUtil.getLanguageByKey(this, "shoot"), ChatFunAdapter.TYPE_CAMERA, R.mipmap.ic_chat_camera));
        this.mFunList.add(new ChatFunModel(LanguageUtil.getLanguageByKey(this, AnalyticsEvents.PARAMETER_SHARE_DIALOG_CONTENT_PHOTO), ChatFunAdapter.TYPE_IMG, R.mipmap.ic_chat_album));
        this.mFunList.add(new ChatFunModel(LanguageUtil.getLanguageByKey(this, "remotePlay"), ChatFunAdapter.TYPE_REMOTE, R.mipmap.ic_chat_remote));
        ChatFunAdapter chatFunAdapter = new ChatFunAdapter(this, this.mFunList);
        this.mFunAdapter = chatFunAdapter;
        chatFunAdapter.setOnItemClickListener(new ChatFunAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.ChatActivity.12
            @Override // com.hismith.hismithapp.adapter.ChatFunAdapter.OnItemClickListener
            public void onItemClick(String str) {
                if (ChatFunAdapter.TYPE_IMG.equals(str)) {
                    ChatActivity.this.sendPictureMsg();
                    return;
                }
                if (ChatFunAdapter.TYPE_CAMERA.equals(str)) {
                    ChatActivity.this.sendCameraMsg();
                    return;
                }
                if (ChatFunAdapter.TYPE_REMOTE.equals(str)) {
                    LogUtil.e("SocketManager isConnect:" + SocketManager.getInstance().isConnect());
                    if (HuDongPlayService.getClientConnection() == 1) {
                        Intent intent = new Intent(ChatActivity.this, (Class<?>) ControlSponsorActivity2.class);
                        intent.addFlags(268435456);
                        ChatActivity.this.startActivity(intent);
                        return;
                    }
                    if (HuDongPlayService.getServerConnection() == 1) {
                        Intent intent2 = new Intent(ChatActivity.this, (Class<?>) ControlAcceptActivity2.class);
                        intent2.addFlags(268435456);
                        ChatActivity.this.startActivity(intent2);
                        return;
                    }
                    int status = BleManager.getInstance(ChatActivity.this).getStatus();
                    ChatActivity chatActivity = ChatActivity.this;
                    if (status != 2) {
                        ToastUtil.showShortToastCenter(ChatActivity.this, LanguageUtil.getLanguageByKey(chatActivity, "noConnectedDevice"));
                    } else {
                        if (chatActivity.progressDialog != null && !ChatActivity.this.progressDialog.isShowing()) {
                            ChatActivity.this.progressDialog.show();
                        }
                        SocketManager.getInstance().connSocket(5);
                    }
                }
            }
        });
        this.mRecyclerFun.setLayoutManager(this.mFunLayoutManager);
        this.mRecyclerFun.setAdapter(this.mFunAdapter);
    }

    private void initMoreDialog() {
        MoreDialog moreDialog = new MoreDialog(this, LanguageUtil.getLanguageByKey(this, "clearChatHistory"), LanguageUtil.getLanguageByKey(this, "report"), this.mUserModel.getIsBlocked() == 1 ? LanguageUtil.getLanguageByKey(this, "removeFromBlocklist") : LanguageUtil.getLanguageByKey(this, "addToBlackList"));
        moreDialog.setOnClickItemActionListener(new MoreDialog.OnClickItemActionListener() { // from class: com.hismith.hismithapp.ChatActivity.10
            @Override // com.hismith.hismithapp.dialog.MoreDialog.OnClickItemActionListener
            public void btnClickOne() {
                ChatActivity.this.clearChatHistoryDB();
            }

            @Override // com.hismith.hismithapp.dialog.MoreDialog.OnClickItemActionListener
            public void btnClickThree() {
                int isBlocked = ChatActivity.this.mUserModel.getIsBlocked();
                ChatActivity chatActivity = ChatActivity.this;
                if (isBlocked == 1) {
                    chatActivity.setUserBlackList(1);
                } else {
                    chatActivity.setUserBlackList(0);
                }
            }

            @Override // com.hismith.hismithapp.dialog.MoreDialog.OnClickItemActionListener
            public void btnClickTwo() {
                new ReportSelectDialog(ChatActivity.this) { // from class: com.hismith.hismithapp.ChatActivity.10.1
                    @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                    public void btnSubmit(String str, String str2) {
                        ChatActivity.this.complaint(str, str2);
                    }
                }.showDialog();
            }
        });
        moreDialog.show();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initRecordPermission() {
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.ChatActivity$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                ChatActivity.this.m555x60682096((Permission) obj);
            }
        });
        if (this.mCompositeDisposable == null || this.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.mCompositeDisposable.add(subscribe);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initSocketData(int i, String str) {
        if (i == 0) {
            try {
                SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.gson.fromJson(str, SocketSpeedModel.class);
                if (socketSpeedModel != null && socketSpeedModel.getSuccess() == 1 && !TextUtils.isEmpty(socketSpeedModel.getGameToken())) {
                    LogUtil.e("连接成功，正在通知对方。。。");
                    msgRemoteSend(socketSpeedModel.getGameToken());
                    return;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        if (this.progressDialog == null || !this.progressDialog.isShowing()) {
            return;
        }
        this.progressDialog.dismiss();
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleImgMore = (ImageView) findViewById(R.id.title_img_more);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.chat_refreshLayout);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.chat_recyclerView);
        this.mRecyclerEmoji = (RecyclerView) findViewById(R.id.chat_recycler_emoji);
        this.mRecyclerFun = (RecyclerView) findViewById(R.id.chat_recycler_fun);
        this.mImgVoice = (ImageView) findViewById(R.id.chat_img_voice);
        this.mImgEmoji = (ImageView) findViewById(R.id.chat_img_emoji);
        this.mImgFun = (ImageView) findViewById(R.id.chat_img_fun);
        this.mEditMessage = (EditText) findViewById(R.id.chat_edit_message);
        this.mTxtVoice = (TextView) findViewById(R.id.chat_txt_voice);
        this.mLayoutVoiceStatus = (LinearLayout) findViewById(R.id.chat_layout_voice_status);
        this.mLayoutEmoji = (LinearLayout) findViewById(R.id.chat_layout_emoji);
        this.mLayoutFun = (LinearLayout) findViewById(R.id.chat_layout_fun);
        this.mImgVoiceStatus = (ImageView) findViewById(R.id.chat_img_voice_status);
        this.mTxtVoiceStatus = (TextView) findViewById(R.id.chat_txt_voice_status);
        this.mReferenceLayout = (ConstraintLayout) findViewById(R.id.chat_reference_layout);
        this.mReferenceContent = (TextView) findViewById(R.id.chat_reference_content);
        this.mReferenceClose = (ImageView) findViewById(R.id.chat_reference_close);
        LanguageUtil.setText(this, this.mTxtVoice, "keepTalking");
        this.mLayoutVoiceStatus.setBackground(ContextCompat.getDrawable(this, R.mipmap.bg_chat_voice));
        this.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_chat_voice_volume_1));
        LanguageUtil.setText(this, this.mTxtVoiceStatus, "moveYourFingerUpCancel");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessage(MessageModel.Message message) {
        try {
            String obj = SharedPreferencesUtil.get(this, Constans.deleteSession).toString();
            if (!TextUtils.isEmpty(obj) && obj.contains("," + message.getUserId() + ",")) {
                SharedPreferencesUtil.put(this, Constans.deleteSession, obj.replace("," + message.getUserId() + ",", ""));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        ContentValues contentValues = new ContentValues();
        this.lastMsgId = message.getMsgId();
        contentValues.put("msgId", Integer.valueOf(message.getMsgId()));
        contentValues.put("userId", Integer.valueOf(message.getUserId()));
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("type", message.getType());
        contentValues.put("voiceLength", Integer.valueOf(message.getVoiceLength()));
        contentValues.put(FirebaseAnalytics.Param.CONTENT, message.getContent());
        contentValues.put("localPath", message.getLocalPath());
        contentValues.put("date", message.getDate());
        contentValues.put("revoke", Integer.valueOf(message.getRevoke()));
        contentValues.put("androidSend", Integer.valueOf(message.getAndroidSend()));
        contentValues.put("androidRead", (Integer) 1);
        contentValues.put("referenceId", Integer.valueOf(message.getReferenceId()));
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE, null, contentValues);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessageLast(MessageModel.Message message) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("msgLastId", Integer.valueOf(message.getMsgId()));
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_LAST, null, contentValues);
    }

    private void insertDBMessageUser(MessageModel.Message message) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("userId", Integer.valueOf(message.getUserId()));
        contentValues.put("king", Integer.valueOf(message.getKing()));
        contentValues.put("gradeIcon", message.getGradeIcon());
        contentValues.put("name", message.getName());
        contentValues.put("gender", Integer.valueOf(message.getGender()));
        contentValues.put("age", message.getAge());
        contentValues.put("constellation", message.getConstellation());
        contentValues.put("headIcon", message.getHeadIcon());
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_USER, null, contentValues);
    }

    static /* synthetic */ Pair lambda$uploadFile$1(String str, UploadModel uploadModel) throws Exception {
        return new Pair(uploadModel, str);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void msgAction(final int i, final int i2) {
        String str;
        String str2 = (String) SharedPreferencesUtil.get(this, Constans.Token);
        if (i2 == 1) {
            str = "delete";
        } else if (i2 == 2) {
            str = "revoke";
        } else if (i2 != 3) {
            return;
        } else {
            str = "revoked";
        }
        this.baseRequest.getApi().msgAction(str2, this.mList.get(i).getMsgId(), str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.ChatActivity.18
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                try {
                    if (httpModel.getSuccess() == 1) {
                        int i3 = i2;
                        if (i3 == 1) {
                            ChatActivity.this.removeChatDB(i);
                        } else if (i3 == 2) {
                            ChatActivity.this.revokeChatDB(i);
                            ChatActivity.this.revokeReferenceDB(i);
                            ((MessageModel.Message) ChatActivity.this.mList.get(i)).setRevoke(1);
                            ChatActivity.this.mAdapter.notifyDataSetChanged();
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void registerActivityResult() {
        this.nameResultLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.ChatActivity.1
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                Intent data;
                ChatActivity chatActivity;
                String str;
                if (activityResult.getResultCode() != 1 || (data = activityResult.getData()) == null) {
                    return;
                }
                String stringExtra = data.getStringExtra("alias");
                if (TextUtils.isEmpty(stringExtra)) {
                    return;
                }
                ChatActivity.this.mTitleTxtName.setText(stringExtra);
                if (ChatActivity.this.mUserModel != null) {
                    ChatActivity.this.mUserModel.setAlias(stringExtra);
                }
                int i = 0;
                while (true) {
                    int size = ChatActivity.this.mList.size();
                    chatActivity = ChatActivity.this;
                    if (i >= size) {
                        break;
                    }
                    if (((MessageModel.Message) chatActivity.mList.get(i)).getReferenceMsg() != null && ((MessageModel.Message) ChatActivity.this.mList.get(i)).getUserId() == ChatActivity.this.mUserModel.getUserId()) {
                        ((MessageModel.Message) ChatActivity.this.mList.get(i)).getReferenceMsg().setUserName(stringExtra);
                    }
                    i++;
                }
                if (chatActivity.mAdapter != null) {
                    ChatActivity.this.mAdapter.notifyDataSetChanged();
                }
                if (ChatActivity.this.mReferenceMessage == null || ChatActivity.this.mReferenceMessage.getAndroidSend() != 0) {
                    return;
                }
                ChatActivity.this.mReferenceMessage.setName(stringExtra);
                boolean equals = "image".equals(ChatActivity.this.mReferenceMessage.getType());
                ChatActivity chatActivity2 = ChatActivity.this;
                if (equals) {
                    str = "[" + LanguageUtil.getLanguageByKey(chatActivity2, "image") + "]";
                } else {
                    boolean equals2 = "voice".equals(chatActivity2.mReferenceMessage.getType());
                    ChatActivity chatActivity3 = ChatActivity.this;
                    if (equals2) {
                        str = "[" + LanguageUtil.getLanguageByKey(chatActivity3, "voice") + "]" + ChatActivity.this.mReferenceMessage.getVoiceLength() + "''";
                    } else {
                        boolean equals3 = "video".equals(chatActivity3.mReferenceMessage.getType());
                        ChatActivity chatActivity4 = ChatActivity.this;
                        str = equals3 ? "[" + LanguageUtil.getLanguageByKey(chatActivity4, "video") + "]" : chatActivity4.mReferenceMessage.getContent();
                    }
                }
                ChatActivity.this.mReferenceContent.setText(ChatActivity.this.mUserModel.getAlias() + ":" + str);
            }
        });
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_MESSAGE_NEW);
        intentFilter.addAction(SocketManager.RECEIVER_SOCKET_NOTIFY);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void removeChatDB(int i) {
        try {
            this.dbWrite.delete(DatabaseHelper.TABLE_MESSAGE, "msgId = ? and sendUserId = ?", new String[]{this.mList.get(i).getMsgId() + "", MyApplication.getApplication().getmUserModel().getUserId() + ""});
            MessageModel.Message message = this.mReferenceMessage;
            if (message != null && message.getMsgId() == this.mList.get(i).getMsgId()) {
                clearReference();
            }
            this.mList.remove(i);
            this.mAdapter.notifyDataSetChanged();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void revokeChatDB(int i) {
        try {
            ContentValues contentValues = new ContentValues();
            contentValues.put("revoke", (Integer) 1);
            this.dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "msgId = ? and sendUserId = ?", new String[]{this.mList.get(i).getMsgId() + "", MyApplication.getApplication().getmUserModel().getUserId() + ""});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void revokeReferenceDB(int i) {
        try {
            this.dbWrite.delete(DatabaseHelper.TABLE_MESSAGE_REFERENCE, "referenceId = ? OR msgId = ?", new String[]{this.mList.get(i).getMsgId() + "", this.mList.get(i).getMsgId() + ""});
            for (MessageModel.Message message : this.mList) {
                if (message.getReferenceMsg() != null && message.getReferenceId() == this.mList.get(i).getMsgId()) {
                    message.setReferenceMsg(null);
                }
            }
            MessageModel.Message message2 = this.mReferenceMessage;
            if (message2 == null || message2.getMsgId() != this.mList.get(i).getMsgId()) {
                return;
            }
            clearReference();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void sendCameraMsg() {
        PictureSelectorManager.getInstance(this).openCamera(new PictureSelectorManager.OnCameraListener() { // from class: com.hismith.hismithapp.ChatActivity.22
            @Override // com.hismith.hismithapp.picker.PictureSelectorManager.OnCameraListener
            public void onCameraResult(List<LocalMedia> list) {
                for (int i = 0; i < list.size(); i++) {
                    if (list.get(i).isCompressed()) {
                        String compressPath = list.get(i).getCompressPath();
                        LogUtil.e("新图片库拍摄压缩后：" + compressPath);
                        ChatActivity.this.sendFileMessage("image", compressPath);
                    } else {
                        ChatActivity.this.sendFileMessage("image", list.get(i).getRealPath());
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void sendFileMessage(String str, String str2) {
        LogUtil.e("添加新消息type：" + str + ", file:" + str2 + ", voiceLength:" + this.voiceLength);
        MessageModel messageModel = new MessageModel();
        Objects.requireNonNull(messageModel);
        MessageModel.Message message = messageModel.new Message();
        message.setConstellation(this.mUserModel.getConstellation());
        message.setLocalPath(str2);
        message.setGender(this.mUserModel.getGender());
        message.setHeadIcon(this.mUserModel.getHeadIcon());
        message.setKing(this.mUserModel.getKing());
        message.setGradeIcon(this.mUserModel.getGradeIcon());
        message.setName(this.mUserModel.getName());
        message.setAlias(this.mUserModel.getAlias());
        message.setDate(zeroZoneTime());
        message.setAndroidSend(1);
        message.setStatus(1);
        message.setType(str);
        message.setVoiceLength(this.voiceLength);
        MessageModel.Message message2 = this.mReferenceMessage;
        if (message2 != null) {
            message.setReferenceId(message2.getMsgId());
            ReferenceMsg referenceMsg = new ReferenceMsg();
            referenceMsg.setReferenceId(this.mReferenceMessage.getMsgId());
            referenceMsg.setType(this.mReferenceMessage.getType());
            referenceMsg.setContent(this.mReferenceMessage.getContent());
            referenceMsg.setVoiceLength(this.mReferenceMessage.getVoiceLength());
            referenceMsg.setLocalPath(this.mReferenceMessage.getLocalPath());
            if (this.mReferenceMessage.getAndroidSend() != 1) {
                referenceMsg.setUserId(this.mReferenceMessage.getUserId());
                referenceMsg.setUserName(this.mReferenceMessage.getName());
            } else if (MyApplication.getApplication().getmUserModel() != null) {
                referenceMsg.setUserId(MyApplication.getApplication().getmUserModel().getUserId());
                referenceMsg.setUserName(MyApplication.getApplication().getmUserModel().getName());
            }
            message.setReferenceMsg(referenceMsg);
            if (this.mReferenceLayout.getVisibility() != 8) {
                this.mReferenceLayout.setVisibility(8);
            }
            this.mReferenceMessage = null;
        }
        this.mList.add(message);
        this.mAdapter.notifyDataSetChanged();
        this.mRecyclerView.smoothScrollToPosition(this.mList.size() - 1);
        if ("voice".equals(str)) {
            str = ShareConstants.WEB_DIALOG_PARAM_MEDIA;
        }
        uploadFile(str, str2, this.mList.size() - 1);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void sendPictureMsg() {
        this.pictureResultManager.openGallery(new PictureResultManager.OnGalleryListener() { // from class: com.hismith.hismithapp.ChatActivity.21
            @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
            public void onGalleryResult(Uri uri) {
                if (uri != null) {
                    ChatActivity.this.sendFileMessage("image", uri.toString());
                }
            }

            @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
            public void onGalleryResultFile(String str) {
                ChatActivity.this.sendFileMessage("image", str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void sendTextMessage(String str, String str2) {
        MessageModel messageModel = new MessageModel();
        Objects.requireNonNull(messageModel);
        MessageModel.Message message = messageModel.new Message();
        message.setConstellation(this.mUserModel.getConstellation());
        message.setContent(str2);
        message.setGender(this.mUserModel.getGender());
        message.setHeadIcon(this.mUserModel.getHeadIcon());
        message.setKing(this.mUserModel.getKing());
        message.setGradeIcon(this.mUserModel.getGradeIcon());
        message.setName(this.mUserModel.getName());
        message.setAlias(this.mUserModel.getAlias());
        message.setDate(zeroZoneTime());
        message.setAndroidSend(1);
        message.setStatus(1);
        message.setType(str);
        MessageModel.Message message2 = this.mReferenceMessage;
        if (message2 != null) {
            message.setReferenceId(message2.getMsgId());
            ReferenceMsg referenceMsg = new ReferenceMsg();
            referenceMsg.setReferenceId(this.mReferenceMessage.getMsgId());
            referenceMsg.setType(this.mReferenceMessage.getType());
            referenceMsg.setContent(this.mReferenceMessage.getContent());
            referenceMsg.setVoiceLength(this.mReferenceMessage.getVoiceLength());
            referenceMsg.setLocalPath(this.mReferenceMessage.getLocalPath());
            if (this.mReferenceMessage.getAndroidSend() != 1) {
                referenceMsg.setUserId(this.mReferenceMessage.getUserId());
                referenceMsg.setUserName(this.mReferenceMessage.getName());
            } else if (MyApplication.getApplication().getmUserModel() != null) {
                referenceMsg.setUserId(MyApplication.getApplication().getmUserModel().getUserId());
                referenceMsg.setUserName(MyApplication.getApplication().getmUserModel().getName());
            }
            message.setReferenceMsg(referenceMsg);
            if (this.mReferenceLayout.getVisibility() != 8) {
                this.mReferenceLayout.setVisibility(8);
            }
            this.mReferenceMessage = null;
        }
        this.mList.add(message);
        this.mAdapter.notifyDataSetChanged();
        this.mRecyclerView.smoothScrollToPosition(this.mList.size() - 1);
        msgSend(str, str2, this.mList.size() - 1);
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mTitleImgMore.setOnClickListener(this);
        this.mImgVoice.setOnClickListener(this);
        this.mImgEmoji.setOnClickListener(this);
        this.mImgFun.setOnClickListener(this);
        this.mReferenceClose.setOnClickListener(this);
        this.mTxtVoice.setOnTouchListener(this.onVoiceTouchListener);
        this.mEditMessage.setOnClickListener(this);
        this.mEditMessage.setOnFocusChangeListener(new View.OnFocusChangeListener() { // from class: com.hismith.hismithapp.ChatActivity.3
            @Override // android.view.View.OnFocusChangeListener
            public void onFocusChange(View view, boolean z) {
                if (z) {
                    if (ChatActivity.this.mLayoutEmoji.getVisibility() == 0) {
                        ChatActivity.this.mLayoutEmoji.setVisibility(8);
                    }
                    if (ChatActivity.this.mLayoutFun.getVisibility() == 0) {
                        ChatActivity.this.mLayoutFun.setVisibility(8);
                        ChatActivity.this.mImgFun.setImageResource(R.mipmap.ic_chat_fun);
                    }
                }
            }
        });
        this.mEditMessage.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.ChatActivity.4
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 4 || ChatActivity.this.mUserModel == null) {
                    return false;
                }
                String obj = ChatActivity.this.mEditMessage.getText().toString();
                if (TextUtils.isEmpty(obj)) {
                    return true;
                }
                ChatActivity.this.mEditMessage.setText("");
                ChatActivity.this.sendTextMessage("text", obj);
                SoftKeyboardUtil.hideSoftKeyboard(ChatActivity.this, textView);
                return true;
            }
        });
        this.mRefreshLayout.setOnRefreshListener(new OnRefreshListener() { // from class: com.hismith.hismithapp.ChatActivity.5
            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                ChatActivity.this.getOldDBMessageForUser();
                ChatActivity.this.mRefreshLayout.finishRefresh();
            }
        });
        this.recorderUtil.setOnMicStatusListener(new RecorderUtil.OnMicStatusListener() { // from class: com.hismith.hismithapp.ChatActivity.6
            @Override // com.hismith.hismithapp.utils.RecorderUtil.OnMicStatusListener
            public void onMicAmplitude(double d) {
                if (ChatActivity.this.isCancelVoice) {
                    return;
                }
                if (d <= 60.0d) {
                    ChatActivity.this.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.ic_chat_voice_volume_1));
                    return;
                }
                if (d <= 70.0d) {
                    ChatActivity.this.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.ic_chat_voice_volume_2));
                    return;
                }
                if (d <= 80.0d) {
                    ChatActivity.this.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.ic_chat_voice_volume_3));
                    return;
                }
                ChatActivity chatActivity = ChatActivity.this;
                if (d <= 90.0d) {
                    chatActivity.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.ic_chat_voice_volume_4));
                } else {
                    chatActivity.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(ChatActivity.this, R.mipmap.ic_chat_voice_volume_5));
                }
            }
        });
        this.mRecyclerView.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.ChatActivity.7
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                KeyboardUtils.hideSoftInput(view);
                return false;
            }
        });
        this.mAdapter.setOnActionClickListener(new ChatAdapter.OnActionClickListener() { // from class: com.hismith.hismithapp.ChatActivity.8
            @Override // com.hismith.hismithapp.adapter.ChatAdapter.OnActionClickListener
            public void onCopy(int i) {
                ClipData newPlainText = ClipData.newPlainText(null, ((MessageModel.Message) ChatActivity.this.mList.get(i)).getContent());
                if (ChatActivity.this.clipboard != null) {
                    ChatActivity.this.clipboard.setPrimaryClip(newPlainText);
                }
            }

            @Override // com.hismith.hismithapp.adapter.ChatAdapter.OnActionClickListener
            public void onDelete(int i) {
                ChatActivity.this.msgAction(i, 1);
            }

            @Override // com.hismith.hismithapp.adapter.ChatAdapter.OnActionClickListener
            public void onReference(int i) {
                String str;
                if (ChatActivity.this.mReferenceLayout.getVisibility() != 0) {
                    ChatActivity.this.mReferenceLayout.setVisibility(0);
                }
                boolean equals = "image".equals(((MessageModel.Message) ChatActivity.this.mList.get(i)).getType());
                ChatActivity chatActivity = ChatActivity.this;
                if (equals) {
                    str = "[" + LanguageUtil.getLanguageByKey(chatActivity, "image") + "]";
                } else {
                    boolean equals2 = "voice".equals(((MessageModel.Message) chatActivity.mList.get(i)).getType());
                    ChatActivity chatActivity2 = ChatActivity.this;
                    if (equals2) {
                        str = "[" + LanguageUtil.getLanguageByKey(chatActivity2, "voice") + "]" + ((MessageModel.Message) ChatActivity.this.mList.get(i)).getVoiceLength() + "''";
                    } else {
                        boolean equals3 = "video".equals(((MessageModel.Message) chatActivity2.mList.get(i)).getType());
                        ChatActivity chatActivity3 = ChatActivity.this;
                        str = equals3 ? "[" + LanguageUtil.getLanguageByKey(chatActivity3, "video") + "]" : ((MessageModel.Message) chatActivity3.mList.get(i)).getContent();
                    }
                }
                int androidSend = ((MessageModel.Message) ChatActivity.this.mList.get(i)).getAndroidSend();
                ChatActivity chatActivity4 = ChatActivity.this;
                if (androidSend == 1) {
                    chatActivity4.mReferenceContent.setText(MyApplication.getApplication().getmUserModel().getName() + ":" + str);
                } else {
                    chatActivity4.mReferenceContent.setText(ChatActivity.this.mUserModel.getAlias() + ":" + str);
                }
                ChatActivity chatActivity5 = ChatActivity.this;
                chatActivity5.mReferenceMessage = (MessageModel.Message) chatActivity5.mList.get(i);
            }

            @Override // com.hismith.hismithapp.adapter.ChatAdapter.OnActionClickListener
            public void onRevoke(int i) {
                ChatActivity.this.msgAction(i, 2);
            }
        });
        this.mAdapter.setClickCallback(new ChatAdapter.ClickCallback() { // from class: com.hismith.hismithapp.ChatActivity.9
            @Override // com.hismith.hismithapp.adapter.ChatAdapter.ClickCallback
            public void onClickUser(int i) {
                Intent intent = new Intent(ChatActivity.this, (Class<?>) UserDetailsActivity2.class);
                intent.putExtra("userId", ((MessageModel.Message) ChatActivity.this.mList.get(i)).getUserId());
                ChatActivity.this.nameResultLauncher.launch(intent);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setUserBlackList(int i) {
        this.baseRequest.getApi().setUserBlackList((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.mUserModel.getUserId()), i).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<BlockModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.ChatActivity.15
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(BlockModel blockModel) {
                if (blockModel.getSuccess() == 1) {
                    ChatActivity.this.mUserModel.setIsBlocked(blockModel.getIsBlocked());
                    Intent intent = new Intent();
                    intent.putExtra("IsBlocked", blockModel.getIsBlocked());
                    ChatActivity.this.setResult(UserDetailsActivity2.REQUEST_USER_INFO, intent);
                    int isBlocked = blockModel.getIsBlocked();
                    ChatActivity chatActivity = ChatActivity.this;
                    if (isBlocked == 1) {
                        ToastUtil.showShortToastCenter(ChatActivity.this, LanguageUtil.getLanguageByKey(chatActivity, "successfullyBlocked"));
                    } else {
                        ToastUtil.showShortToastCenter(ChatActivity.this, LanguageUtil.getLanguageByKey(chatActivity, "removed"));
                    }
                }
            }
        });
    }

    private void startRecord() {
        this.mLayoutVoiceStatus.setVisibility(0);
        this.isCancelVoice = false;
        this.mLayoutVoiceStatus.setBackground(ContextCompat.getDrawable(this, R.mipmap.bg_chat_voice));
        this.mImgVoiceStatus.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_chat_voice_volume_1));
        LanguageUtil.setText(this, this.mTxtVoiceStatus, "moveYourFingerUpCancel");
        this.recorderUtil.startRecording("VOICE_" + String.valueOf(System.currentTimeMillis()) + PictureMimeType.MP4);
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void updateDBManagerRead() {
        if (this.mUserModel == null) {
            return;
        }
        try {
            ContentValues contentValues = new ContentValues();
            contentValues.put("androidRead", (Integer) 1);
            this.dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "userId = ? and sendUserId = ?", new String[]{this.mUserModel.getUserId() + "", MyApplication.getApplication().getmUserModel().getUserId() + ""});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void updateStatus(int i, int i2) {
        this.mList.get(i).setStatus(i2);
        this.mAdapter.notifyItemChanged(i, "");
    }

    /* renamed from: lambda$initRecordPermission$3$com-hismith-hismithapp-ChatActivity, reason: not valid java name */
    /* synthetic */ void m555x60682096(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            if (this.action == 0) {
                startRecord();
                return;
            }
            return;
        }
        if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "microPermission"));
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
        }
    }

    /* renamed from: lambda$uploadFile$0$com-hismith-hismithapp-ChatActivity, reason: not valid java name */
    /* synthetic */ String m556lambda$uploadFile$0$comhismithhismithappChatActivity(String str, String str2) throws Exception {
        String compressImage;
        return (!"image".equals(str) || (compressImage = ImageCompressor.compressImage(this, str2, 1080, 200)) == null) ? str2 : compressImage;
    }

    /* renamed from: lambda$uploadFile$2$com-hismith-hismithapp-ChatActivity, reason: not valid java name */
    /* synthetic */ ObservableSource m557lambda$uploadFile$2$comhismithhismithappChatActivity(String str, String str2, final String str3) throws Exception {
        Map<String, RequestBody> buildUploadParams = buildUploadParams(str, str3);
        if (str3.equals(str2)) {
            str3 = null;
        }
        return this.fileRequest.getApi().uploadFile(buildUploadParams).map(new Function() { // from class: com.hismith.hismithapp.ChatActivity$$ExternalSyntheticLambda1
            @Override // io.reactivex.functions.Function
            public final Object apply(Object obj) {
                return ChatActivity.lambda$uploadFile$1(str3, (UploadModel) obj);
            }
        });
    }

    public void msgRemoteSend(final String str) {
        this.baseRequest.getApi().msgSend((String) SharedPreferencesUtil.get(this, Constans.Token), String.valueOf(System.currentTimeMillis()), Integer.valueOf(this.mUserModel.getUserId()), str, 0, Constans.MESSAGE_TYPE_REMOTE).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<MsgSendModel>(this, null, false) { // from class: com.hismith.hismithapp.ChatActivity.17
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                LogUtil.e("通知已发送，正在对待对方响应");
                if (ChatActivity.this.progressDialog == null || !ChatActivity.this.progressDialog.isShowing()) {
                    return;
                }
                ChatActivity.this.progressDialog.dismiss();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                LogUtil.e("网络异常，通知发送失败。请挂断重新发出");
                if (ChatActivity.this.progressDialog == null || !ChatActivity.this.progressDialog.isShowing()) {
                    return;
                }
                ChatActivity.this.progressDialog.dismiss();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(MsgSendModel msgSendModel) {
                String str2;
                String str3;
                if (msgSendModel.getSuccess() == 1) {
                    MessageModel messageModel = new MessageModel();
                    Objects.requireNonNull(messageModel);
                    MessageModel.Message message = messageModel.new Message();
                    if (ChatActivity.this.mUserModel != null) {
                        String name = ChatActivity.this.mUserModel.getName();
                        String headIcon = ChatActivity.this.mUserModel.getHeadIcon();
                        message.setGender(ChatActivity.this.mUserModel.getGender());
                        message.setHeadIcon(ChatActivity.this.mUserModel.getHeadIcon());
                        message.setKing(ChatActivity.this.mUserModel.getKing());
                        message.setGradeIcon(ChatActivity.this.mUserModel.getGradeIcon());
                        message.setName(ChatActivity.this.mUserModel.getName());
                        message.setAlias(ChatActivity.this.mUserModel.getAlias());
                        message.setConstellation(ChatActivity.this.mUserModel.getConstellation());
                        str2 = name;
                        str3 = headIcon;
                    } else {
                        str2 = "";
                        str3 = str2;
                    }
                    message.setAndroidSend(1);
                    message.setContent(msgSendModel.getContent());
                    message.setDate(msgSendModel.getDate());
                    message.setMsgId(msgSendModel.getMsgId());
                    message.setRevoke(0);
                    message.setType(msgSendModel.getType());
                    message.setUserId(msgSendModel.getUserId());
                    message.setVoiceLength(msgSendModel.getVoiceLength());
                    message.setStatus(0);
                    message.setReferenceId(msgSendModel.getReferenceId());
                    if (msgSendModel.getMsgId() > 0) {
                        DBMessageUtil.insertDBMessage(ChatActivity.this, message);
                    }
                    DBMessageUtil.insertDBMessageLast(message);
                    ChatActivity.this.mList.add(message);
                    ChatActivity.this.mAdapter.notifyDataSetChanged();
                    ChatActivity.this.mRecyclerView.smoothScrollToPosition(ChatActivity.this.mList.size() - 1);
                    HuDongPlayService.startService(ChatActivity.this, 0, message.getMsgId(), str2, str3, str);
                }
            }
        });
    }

    public void msgSend(String str, String str2, final int i) {
        String str3 = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String valueOf = String.valueOf(System.currentTimeMillis());
        this.mList.get(i).setContent(str2);
        this.baseRequest.getApi().msgSend(str3, valueOf, Integer.valueOf(this.mUserModel.getUserId()), str2, Integer.valueOf(this.mList.get(i).getVoiceLength()), str, Integer.valueOf(this.mList.get(i).getReferenceId())).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<MsgSendModel>(this, null, false) { // from class: com.hismith.hismithapp.ChatActivity.16
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((MessageModel.Message) ChatActivity.this.mList.get(i)).setStatus(2);
                ChatActivity.this.mAdapter.notifyItemChanged(i, "");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(MsgSendModel msgSendModel) {
                if (msgSendModel.getSuccess() != 1) {
                    if (msgSendModel.getSuccess() == 0) {
                        ((MessageModel.Message) ChatActivity.this.mList.get(i)).setStatus(2);
                        ChatActivity.this.mAdapter.notifyItemChanged(i, "");
                        return;
                    }
                    return;
                }
                MessageModel.Message message = (MessageModel.Message) ChatActivity.this.mList.get(i);
                message.setContent(msgSendModel.getContent());
                message.setDate(msgSendModel.getDate());
                message.setMsgId(msgSendModel.getMsgId());
                message.setRevoke(0);
                message.setAndroidRead(1);
                message.setType(msgSendModel.getType());
                message.setUserId(msgSendModel.getUserId());
                message.setVoiceLength(msgSendModel.getVoiceLength());
                message.setStatus(0);
                message.setReferenceId(msgSendModel.getReferenceId());
                if (msgSendModel.getMsgId() > 0) {
                    ChatActivity.this.insertDBMessage(message);
                }
                ChatActivity.this.insertDBMessageLast(message);
                if (message.getReferenceId() > 0 && message.getReferenceMsg() != null) {
                    DBMessageUtil.insertMessageReference(msgSendModel.getMsgId(), message.getReferenceMsg());
                }
                ChatActivity.this.mList.set(i, message);
                ChatActivity.this.mAdapter.notifyItemChanged(i, "");
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.title_img_more) {
            initMoreDialog();
            return;
        }
        if (id == R.id.chat_edit_message) {
            if (this.mLayoutEmoji.getVisibility() == 0) {
                this.mLayoutEmoji.setVisibility(8);
            }
            if (this.mLayoutFun.getVisibility() == 0) {
                this.mLayoutFun.setVisibility(8);
                this.mImgFun.setImageResource(R.mipmap.ic_chat_fun);
                return;
            }
            return;
        }
        if (id == R.id.chat_img_voice) {
            boolean z = this.isSendText;
            ImageView imageView = this.mImgVoice;
            if (z) {
                imageView.setImageDrawable(ContextCompat.getDrawable(this, R.drawable.selector_chat_keyboard));
                this.mEditMessage.setVisibility(8);
                this.mTxtVoice.setVisibility(0);
                if (this.mLayoutEmoji.getVisibility() == 0) {
                    this.mLayoutEmoji.setVisibility(8);
                }
                if (this.mLayoutFun.getVisibility() == 0) {
                    this.mLayoutFun.setVisibility(8);
                    this.mImgFun.setImageResource(R.mipmap.ic_chat_fun);
                }
            } else {
                imageView.setImageDrawable(ContextCompat.getDrawable(this, R.drawable.selector_chat_voice));
                this.mTxtVoice.setVisibility(8);
                this.mEditMessage.setVisibility(0);
            }
            this.isSendText = !this.isSendText;
            return;
        }
        if (id == R.id.chat_img_emoji) {
            int visibility = this.mLayoutEmoji.getVisibility();
            LinearLayout linearLayout = this.mLayoutEmoji;
            if (visibility == 0) {
                linearLayout.setVisibility(8);
                return;
            }
            linearLayout.setVisibility(0);
            if (this.mLayoutFun.getVisibility() == 0) {
                this.mLayoutFun.setVisibility(8);
                this.mImgFun.setImageResource(R.mipmap.ic_chat_fun);
            }
            SoftKeyboardUtil.hideSoftKeyboard(this, view);
            if (this.isSendText) {
                return;
            }
            this.mImgVoice.setImageDrawable(ContextCompat.getDrawable(this, R.drawable.selector_chat_voice));
            this.mTxtVoice.setVisibility(8);
            this.mEditMessage.setVisibility(0);
            this.isSendText = true;
            return;
        }
        if (id != R.id.chat_img_fun) {
            if (id == R.id.chat_reference_close) {
                clearReference();
                return;
            }
            return;
        }
        int visibility2 = this.mLayoutFun.getVisibility();
        LinearLayout linearLayout2 = this.mLayoutFun;
        if (visibility2 == 0) {
            linearLayout2.setVisibility(8);
            this.mImgFun.setImageResource(R.mipmap.ic_chat_fun);
            return;
        }
        linearLayout2.setVisibility(0);
        this.mImgFun.setImageResource(R.mipmap.ic_chat_fun_selected);
        if (this.mLayoutEmoji.getVisibility() == 0) {
            this.mLayoutEmoji.setVisibility(8);
        }
        SoftKeyboardUtil.hideSoftKeyboard(this, view);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_chat);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        closeDB();
        unRegisterReceiver();
        this.mHandler.removeCallbacksAndMessages(null);
        this.recorderUtil.cancelRecording();
        setResult(PersonalMessageFragment.REQUEST_MESSAGE_NEW);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onPause() {
        super.onPause();
        try {
            if (this.mEditMessage.hasFocus()) {
                this.mEditMessage.clearFocus();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void uploadFile(final String str, final String str2, final int i) {
        try {
            if (TextUtils.isEmpty(str2)) {
                updateStatus(i, 3);
                return;
            }
            if (!PictureMimeType.isContent(str2)) {
                File file = new File(str2);
                if (!file.exists() || file.isDirectory()) {
                    updateStatus(i, 3);
                    return;
                }
            }
            Disposable subscribe = Observable.just(str2).subscribeOn(Schedulers.io()).map(new Function() { // from class: com.hismith.hismithapp.ChatActivity$$ExternalSyntheticLambda2
                @Override // io.reactivex.functions.Function
                public final Object apply(Object obj) {
                    return ChatActivity.this.m556lambda$uploadFile$0$comhismithhismithappChatActivity(str, (String) obj);
                }
            }).flatMap(new Function() { // from class: com.hismith.hismithapp.ChatActivity$$ExternalSyntheticLambda3
                @Override // io.reactivex.functions.Function
                public final Object apply(Object obj) {
                    return ChatActivity.this.m557lambda$uploadFile$2$comhismithhismithappChatActivity(str, str2, (String) obj);
                }
            }).observeOn(AndroidSchedulers.mainThread()).subscribe(new Consumer<Pair<UploadModel, String>>() { // from class: com.hismith.hismithapp.ChatActivity.19
                @Override // io.reactivex.functions.Consumer
                public void accept(Pair<UploadModel, String> pair) throws Exception {
                    int success = ((UploadModel) pair.first).getSuccess();
                    ChatActivity chatActivity = ChatActivity.this;
                    if (success == 1) {
                        chatActivity.msgSend(((MessageModel.Message) chatActivity.mList.get(i)).getType(), ((UploadModel) pair.first).getFileUrl(), i);
                    } else {
                        chatActivity.updateStatus(i, 3);
                    }
                    if (pair.second != null) {
                        new File((String) pair.second).delete();
                    }
                }
            }, new Consumer<Throwable>() { // from class: com.hismith.hismithapp.ChatActivity.20
                @Override // io.reactivex.functions.Consumer
                public void accept(Throwable th) throws Exception {
                    th.printStackTrace();
                    ChatActivity.this.updateStatus(i, 3);
                }
            });
            if (this.mCompositeDisposable != null) {
                this.mCompositeDisposable.add(subscribe);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String zeroZoneTime() {
        this.sdf.setTimeZone(TimeZone.getTimeZone("GMT+00:00"));
        return this.sdf.format(new Date());
    }
}
