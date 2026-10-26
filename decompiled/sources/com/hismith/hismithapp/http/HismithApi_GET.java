package com.hismith.hismithapp.http;

import com.hismith.hismithapp.model.BarrageModel;
import com.hismith.hismithapp.model.BlackModel;
import com.hismith.hismithapp.model.BlockModel;
import com.hismith.hismithapp.model.ChatUpModel;
import com.hismith.hismithapp.model.CheckUpdateModel;
import com.hismith.hismithapp.model.ComplaintReasonsModel;
import com.hismith.hismithapp.model.CountryModel;
import com.hismith.hismithapp.model.FansFocusModel;
import com.hismith.hismithapp.model.FirmwareModel;
import com.hismith.hismithapp.model.FollowModel;
import com.hismith.hismithapp.model.InteractiveModel;
import com.hismith.hismithapp.model.LikeUserModel;
import com.hismith.hismithapp.model.LoginModel;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.ModeBlockModel;
import com.hismith.hismithapp.model.MsgSendModel;
import com.hismith.hismithapp.model.PlayLogsListModel;
import com.hismith.hismithapp.model.PlayLogsModel;
import com.hismith.hismithapp.model.PlayModeActionModel;
import com.hismith.hismithapp.model.PlayModeCreateModel;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.ProductsModel;
import com.hismith.hismithapp.model.ReviewDetailsModel;
import com.hismith.hismithapp.model.ReviewLikeModel;
import com.hismith.hismithapp.model.ReviewSendModel;
import com.hismith.hismithapp.model.TokenModel;
import com.hismith.hismithapp.model.TopicActionModel;
import com.hismith.hismithapp.model.TopicCategoriesModel;
import com.hismith.hismithapp.model.TopicListModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.model.TopicReviewModel;
import com.hismith.hismithapp.model.TopicSendModel;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.model.UserSearchModel;
import com.hismith.hismithapp.utils.Constans;
import io.reactivex.Observable;
import java.util.Map;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.PartMap;
import retrofit2.http.Query;
import retrofit2.http.Streaming;
import retrofit2.http.Url;

/* loaded from: classes3.dex */
public interface HismithApi_GET {
    @GET("allProducts_v1")
    Observable<ProductsModel> allProducts_v1();

    @GET("changeMail")
    Observable<HttpModel> changeMail(@Query("token") String str, @Query("email") String str2, @Query("code") Integer num);

    @GET("changePass")
    Observable<HttpModel> changePass(@Query("token") String str, @Query("oldPassword") String str2, @Query("password") String str3);

    @GET("checkUpdate")
    Observable<CheckUpdateModel> checkUpdate(@Query("token") String str, @Query("allLangs") String str2, @Query("dev") Integer num);

    @GET("checkUpdate")
    Observable<CheckUpdateModel> checkUpdate(@Query("token") String str, @Query("allLangs") String str2, @Query("dev") Integer num, @Query("type") String str3, @Query("version") String str4);

    @GET("checkUpdate")
    Observable<CheckUpdateModel> checkUpdate(@Query("token") String str, @Query("allLangs") String str2, @Query("dev") Integer num, @Query("type") String str3, @Query("version") String str4, @Query("ads") String str5);

    @GET("complaint")
    Observable<HttpModel> complaint(@Query("token") String str, @Query("id") Integer num, @Query("type") String str2, @Query("reason") String str3, @Query("content") String str4);

    @GET(Constans.complaintReasons)
    Observable<ComplaintReasonsModel> complaintReasons(@Query("token") String str);

    @Streaming
    @POST
    Observable<ResponseBody> download(@Url String str);

    @GET("emailVerCode")
    Observable<HttpModel> emailVerCode(@Query("type") String str, @Query("email") String str2);

    @GET("emailVerCode")
    Observable<HttpModel> emailVerCode(@Query("token") String str, @Query("type") String str2, @Query("email") String str3);

    @GET("firmwareVersion")
    Observable<FirmwareModel> firmwareVersion(@Query("token") String str, @Query("action") String str2, @Query("productId") String str3);

    @GET("follow")
    Observable<FollowModel> follow(@Query("token") String str, @Query("userId") Integer num, @Query("follow") Integer num2);

    @GET("forgotPass")
    Observable<LoginModel> forgotPass(@Query("email") String str, @Query("password") String str2, @Query("code") Integer num);

    @GET("getCountry")
    Observable<CountryModel> getCountry();

    @GET("getFansFocus")
    Observable<FansFocusModel> getFansFocus(@Query("token") String str, @Query("mode") String str2, @Query("page") Integer num);

    @GET("getUserInfo")
    Observable<UserModel> getMyInfo(@Query("token") String str, @Query("androidToken") String str2, @Query("appVersion") String str3);

    @GET("playModeBarrage?action=get")
    Observable<BarrageModel> getPlayModeBarrage(@Query("token") String str, @Query("id") int i);

    @GET("getProduct")
    Observable<ProductModel> getProduct(@Query("token") String str, @Query("code") String str2);

    @GET("getUserBlackList")
    Observable<BlackModel> getUserBlackList(@Query("token") String str, @Query("page") Integer num);

    @GET("getUserInfo")
    Observable<UserModel> getUserInfo(@Query("token") String str, @Query("userId") Integer num);

    @GET("userLogin")
    Observable<LoginModel> login(@Query("userName") String str, @Query("userPass") String str2);

    @GET("userLogin")
    Observable<LoginModel> login(@Query("userName") String str, @Query("userPass") String str2, @Query("loginType") String str3);

    @GET("messageNew")
    Observable<MessageModel> messageNew(@Query("token") String str, @Query("lastMsgId") Integer num);

    @GET("messageNew_v1")
    Observable<MessageModel> messageNew_v1(@Query("token") String str, @Query("lastMsgId") Integer num);

    @GET("msgAction")
    Observable<HttpModel> msgAction(@Query("token") String str, @Query("msgId") int i, @Query("action") String str2);

    @GET("msgFirstChat")
    Observable<ChatUpModel> msgFirstChat(@Query("token") String str, @Query("page") Integer num);

    @GET("msgInteractive")
    Observable<InteractiveModel> msgInteractive(@Query("token") String str, @Query("page") Integer num);

    @GET("msgSend")
    Observable<MsgSendModel> msgSend(@Query("token") String str, @Query("localId") String str2, @Query("userId") Integer num, @Query("content") String str3, @Query("voiceLength") Integer num2, @Query("type") String str4);

    @GET("msgSend")
    Observable<MsgSendModel> msgSend(@Query("token") String str, @Query("localId") String str2, @Query("userId") Integer num, @Query("content") String str3, @Query("voiceLength") Integer num2, @Query("type") String str4, @Query("referenceId") Integer num3);

    @GET("userTopic")
    Observable<TopicListModel> myTopic(@Query("token") String str, @Query("type") String str2, @Query("page") Integer num);

    @GET("playLogs")
    Observable<PlayLogsListModel> playLogs(@Query("token") String str, @Query("new") int i, @Query("requestId") String str2);

    @GET("playModeAction")
    Observable<PlayModeActionModel> playModeAction(@Query("token") String str, @Query("id") int i, @Query("action") String str2);

    @GET("playModeAction")
    Observable<PlayModeActionModel> playModeAction(@Query("token") String str, @Query("id") int i, @Query("action") String str2, @Query("name") String str3);

    @GET("playModeBlock")
    Observable<ModeBlockModel> playModeBlock(@Query("token") String str, @Query("requestId") String str2);

    @GET("playModeList")
    Observable<PlayModeListModel> playModeList(@Query("token") String str, @Query("type") String str2, @Query("new") int i, @Query("requestId") String str3, @Query("sortModeTime") String str4, @Query("sortPlayCount") String str5, @Query("filterOnlyThisProduct") int i2, @Query("filterModeName") String str6, @Query("filterModeTime") String str7, @Query("filterModeSpeed") int i3, @Query("filterModePlayCount") String str8, @Query("filterPublishDate") String str9);

    @GET("playModeList")
    Observable<PlayModeListModel> playModeList(@Query("token") String str, @Query("type") String str2, @Query("code") String str3, @Query("new") int i, @Query("requestId") String str4);

    @GET("playModeList")
    Observable<PlayModeListModel> playModeList(@Query("token") String str, @Query("type") String str2, @Query("code") String str3, @Query("new") int i, @Query("requestId") String str4, @Query("sortModeTime") String str5, @Query("sortPlayCount") String str6, @Query("filterOnlyThisProduct") int i2, @Query("filterModeName") String str7, @Query("filterModeTime") String str8, @Query("filterModeSpeed") int i3, @Query("filterModePlayCount") String str9, @Query("filterPublishDate") String str10);

    @GET("playModeList")
    Observable<PlayModeListModel> playModeList(@Query("token") String str, @Query("type") String str2, @Query("userId") String str3, @Query("code") String str4, @Query("new") int i, @Query("requestId") String str5);

    @GET("playModeSave")
    Observable<PlayModeCreateModel> playModeSave(@Query("token") String str, @Query("id") String str2, @Query("name") String str3, @Query("code") String str4, @Query("modeData") String str5);

    @GET("playModeSave")
    Observable<PlayModeCreateModel> playModeSave(@Query("token") String str, @Query("id") String str2, @Query("name") String str3, @Query("code") String str4, @Query("setting") String str5, @Query("modeData") String str6);

    @GET
    Observable<HttpModel> qrcodeLogin(@Url String str, @Query("token") String str2);

    @GET("userRegister")
    Observable<LoginModel> register(@Query("email") String str, @Query("password") String str2, @Query("code") Integer num);

    @GET("removeAccount")
    Observable<HttpModel> removeAccount(@Query("token") String str, @Query("code") String str2);

    @GET("follow")
    Observable<FollowModel> removeFan(@Query("token") String str, @Query("userId") Integer num, @Query("follow") Integer num2, @Query("remove") Integer num3);

    @GET("setUserBlackList")
    Observable<BlockModel> removeUserBlackList(@Query("token") String str, @Query("userId") Integer num, @Query("remove") int i);

    @GET("setUserBlackList")
    Observable<BlockModel> removeUserBlackList(@Query("token") String str, @Query("userId") String str2, @Query("remove") int i);

    @FormUrlEncoded
    @POST("oauth2/v4/token")
    Observable<TokenModel> requestToken(@Field("client_id") String str, @Field("client_secret") String str2, @Field("code") String str3, @Field("redirect_uri") String str4, @Field("grant_type") String str5);

    @FormUrlEncoded
    @POST("oauth2/v4/token")
    Observable<TokenModel> requestTokenRefresh(@Field("refresh_token") String str, @Field("client_id") String str2, @Field("client_secret") String str3, @Field("grant_type") String str4);

    @GET
    Observable<ResponseBody> resetServer(@Url String str);

    @GET("setUserInfo")
    Observable<HttpModel> setChatUpState(@Query("token") String str, @Query("chatUpState") Integer num);

    @GET("setUserInfo")
    Observable<HttpModel> setPlayLogs(@Query("token") String str, @Query("playLogs") Integer num);

    @GET("setPlayLogs")
    Observable<PlayLogsModel> setPlayLogs(@Query("token") String str, @Query("productId") String str2, @Query("gameOver") int i);

    @GET("playModeBarrage?action=set")
    Observable<BarrageModel> setPlayModeBarrage(@Query("token") String str, @Query("id") int i, @Query("second") double d, @Query("color") String str2, @Query("content") String str3);

    @GET("setUserInfo")
    Observable<HttpModel> setRemind(@Query("token") String str, @Query("mutuallyRemind") Integer num, @Query("focusRemind") Integer num2, @Query("messageRemind") Integer num3);

    @GET("setUserInfo")
    Observable<HttpModel> setRemind(@Query("token") String str, @Query("mutuallyRemind") Integer num, @Query("focusRemind") Integer num2, @Query("messageRemind") Integer num3, @Query("productAds") Integer num4);

    @GET("setUserInfo")
    Observable<HttpModel> setStealth(@Query("token") String str, @Query("stealth") Integer num);

    @GET("setUserAlias")
    Observable<HttpModel> setUserAlias(@Query("token") String str, @Query("userId") Integer num, @Query("userAlias") String str2);

    @GET("setUserInfo")
    Observable<HttpModel> setUserBIO(@Query("token") String str, @Query("bios") String str2);

    @GET("setUserBlackList")
    Observable<BlockModel> setUserBlackList(@Query("token") String str, @Query("userId") Integer num);

    @GET("setUserBlackList")
    Observable<BlockModel> setUserBlackList(@Query("token") String str, @Query("userId") Integer num, @Query("remove") int i);

    @GET("setUserInfo")
    Observable<HttpModel> setUserHead(@Query("token") String str, @Query("headIcon") String str2);

    @GET("setUserInfo")
    Observable<HttpModel> setUserInfo(@Query("token") String str, @Query("gender") Integer num, @Query("name") String str2, @Query("countryId") Integer num2, @Query("cityId") Integer num3, @Query("phone") String str3, @Query("birthday") String str4, @Query("headIcon") String str5, @Query("voice") String str6, @Query("voiceLength") Integer num4, @Query("photos") String str7, @Query("chatUpState") Integer num5, @Query("focusRemind") Integer num6, @Query("messageRemind") Integer num7, @Query("mutuallyRemind") Integer num8, @Query("addFriendRemind") Integer num9);

    @GET("setUserInfo")
    Observable<HttpModel> setUserInfo(@Query("token") String str, @Query("photos") String str2);

    @GET("setUserInfo")
    Observable<HttpModel> setUserInfo(@Query("token") String str, @Query("name") String str2, @Query("gender") Integer num, @Query("countryId") Integer num2, @Query("cityId") Integer num3, @Query("birthday") String str3, @Query("phone") String str4, @Query("headIcon") String str5);

    @GET("setUserInfo")
    Observable<HttpModel> setUserInfo(@Query("token") String str, @Query("name") String str2, @Query("countryId") Integer num, @Query("cityId") Integer num2, @Query("birthday") String str3, @Query("gender") Integer num3, @Query("interestedIn") Integer num4, @Query("phone") String str4);

    @GET("setUserLang")
    Observable<HttpModel> setUserLang(@Query("token") String str, @Query("userLang") String str2);

    @GET("setUserInfo")
    Observable<HttpModel> setViewSex(@Query("token") String str, @Query("viewSex") int i);

    @GET("setUserInfo")
    Observable<HttpModel> setVoice(@Query("token") String str, @Query("voice") String str2, @Query("voiceLength") Integer num);

    @GET("submitError")
    Observable<HttpModel> submitError(@Query("token") String str, @Query("app") String str2, @Query("appVersion") String str3, @Query("message") String str4);

    @GET("topicAction")
    Observable<TopicActionModel> topicAction(@Query("token") String str, @Query("id") Integer num, @Query("action") String str2);

    @GET("topicCategories")
    Observable<TopicCategoriesModel> topicCategories(@Query("token") String str);

    @GET("topicDel")
    Observable<HttpModel> topicDel(@Query("token") String str, @Query("id") Integer num);

    @GET("topicDetails")
    Observable<TopicModel> topicDetails(@Query("token") String str, @Query("id") Integer num);

    @GET("topicSend")
    Observable<TopicSendModel> topicEdit(@Query("token") String str, @Query("id") Integer num, @Query("localId") String str2, @Query("title") String str3, @Query("content") String str4);

    @GET("topicList")
    Observable<TopicListModel> topicList(@Query("token") String str, @Query("key") String str2, @Query("new") Integer num, @Query("requestId") String str3);

    @GET("topicList")
    Observable<TopicListModel> topicList(@Query("token") String str, @Query("key") String str2, @Query("categorieId") String str3, @Query("new") Integer num, @Query("requestId") String str4);

    @GET("topicReviewDel")
    Observable<HttpModel> topicReviewDel(@Query("token") String str, @Query("id") Integer num);

    @GET("topicReviewDetails")
    Observable<ReviewDetailsModel> topicReviewDetails(@Query("token") String str, @Query("id") Integer num, @Query("page") Integer num2);

    @GET("topicReviewLike")
    Observable<ReviewLikeModel> topicReviewLike(@Query("token") String str, @Query("id") Integer num, @Query("action") String str2);

    @GET("topicReviewLikeUser")
    Observable<LikeUserModel> topicReviewLikeUser(@Query("token") String str, @Query("id") Integer num, @Query("page") Integer num2);

    @GET("topicReviewSend")
    Observable<ReviewSendModel> topicReviewReplySend(@Query("token") String str, @Query("id") Integer num, @Query("content") String str2, @Query("replyId") Integer num2);

    @GET("topicReviewSend")
    Observable<ReviewSendModel> topicReviewSend(@Query("token") String str, @Query("id") Integer num, @Query("content") String str2);

    @GET("topicReviews")
    Observable<TopicReviewModel> topicReviews(@Query("token") String str, @Query("id") Integer num, @Query("page") Integer num2);

    @GET("topicSend")
    Observable<TopicSendModel> topicSend(@Query("token") String str, @Query("localId") String str2, @Query("categorieId") String str3, @Query("title") String str4, @Query("content") String str5);

    @GET("topicSend")
    Observable<TopicSendModel> topicSend(@Query("token") String str, @Query("localId") String str2, @Query("categorieId") String str3, @Query("tag") String str4, @Query("title") String str5, @Query("content") String str6);

    @POST("uploadFile")
    @Multipart
    Observable<UploadModel> uploadFile(@Field("token") String str, @Field("action") String str2, @Field("fileType") String str3, @Field("requestId") String str4, @PartMap Map<String, RequestBody> map);

    @POST("uploadFile")
    @Multipart
    Observable<UploadModel> uploadFile(@PartMap Map<String, RequestBody> map);

    @GET("userOauthLogin")
    Observable<LoginModel> userOauthLogin(@Query("source") String str, @Query("sourceToken") String str2);

    @GET("userOauthLogin")
    Observable<LoginModel> userOauthLogin(@Query("source") String str, @Query("sourceToken") String str2, @Query("version") String str3);

    @GET("userSearch")
    Observable<UserSearchModel> userSearch(@Query("token") String str, @Query("key") String str2, @Query("page") Integer num);

    @GET("userTopic")
    Observable<TopicListModel> userTopic(@Query("token") String str, @Query("userId") String str2, @Query("page") Integer num);
}
