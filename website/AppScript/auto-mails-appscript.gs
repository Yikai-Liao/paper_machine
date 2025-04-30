/**
 * This script is used to send automated emails to the lab members for the (semi-)weekly lab talks.
 * The script also sends out reminders for abstract submission, talk announcements, and cancellations to the speakers & lab members.
 * The script is run every hour.
 * NOTE: This script can only be tested/ran from the Google Apps Script Editor, not locally.
 */

/**
 * AbsMail: Abstract Submission Mail - Sent 7 days before the talk
 * AbsRem: Abstract Submission Reminder - Sent 3 days before the talk
 * TalkAnn: Talk Announcement - Sent 2 days before the talk
 * TalkCan: Talk Cancelled - Sent if the talk is cancelled, e.g., if abstract is not submitted at least 2 days before the talk
 * TalkRem1: Talk Reminder 1 - Sent on the day of the talk
 * TalkRem2: Talk Reminder 2 - Sent 1 hour before the talk
 */

// FOR DEBUGGING
var pauseMails = false;
var debugCal = false;
var testCalID = "" // CHANGEME: NOTE: Calendar ID for testing

// Google Sheet connection
var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
var sheet = spreadsheet.getSheetByName('Talk Data');
var data = sheet.getDataRange().getValues();

// Venue of the talk
// "E2 (SMLab), near Bank Building" | "CC-03, SMS"
var talkVenue = "E2 (SMLab), near Bank Building"
// Time of the talk
var talkTime = 15.75 // 15:45
// Error string to compare empty fields with
// Applicable to `title, abstract, extra, online`
// TODO: Handle this in Sheets itself perhaps 
var errStr = "#N/A"

// Different times to trigger different emails
var absTrig = 7 * 24 - talkTime; // ~7 days in hours
var absRemTrig = 3 * 24 - talkTime; // ~3 days in hours
var talkAnnTrig = 2 * 24 - talkTime; // ~2 days in hours
var talkRem1Trig = talkTime; // in hours | Reminder sent at 00:00 on the day of the talk
var talkRem2Trig = 1; // 1 hour

var talkGH = "https://github.com/JeS24/smlab-talks/"
var talkSite = "https://smlab.niser.ac.in/labtalks/"
var presEmail = "" // CHANGEME: NOTE: Email ID of the presentation account

// Calendar ID for the lab calendar
var sLabCalID = "" // CHANGEME: NOTE: Calendar ID for the lab calendar
// Google Form link for abstract submission
var formLink = "" // CHANGEME: NOTE: Google Form link for abstract submission
// Google Meet link for the talk
var gMeetInfo = `
To join the video meeting, click this link: https://meet.google.com/xxx-xxxx-xxx
Otherwise, to join by phone, dial +1 984-999-0476 and enter this PIN: 538 765 952#
` // CHANGEME: NOTE: Google Meet link for the talk
var sign = `
Best regards,<br>
<a href="https://github.com/JeS24" target="_blank">Jyotirmaya Shivottam</a><br>
PhD Scholar<br>
<a href="https://www.niser.ac.in/~smishra/" target="_blank">Subhankar Mishra Lab</a><br>
<a href="https://www.niser.ac.in/scos/" target="_blank">School of Computer Sciences</a><br>
<a href="https://www.niser.ac.in/" target="_blank"> National Institute of Science Education and Research</a><br>
An OCC of Homi Bhabha National Institute<br>
Bhubaneswar, India<br><br>

<b>P.S.1</b>: You can mail the slides (any format) to <a href="mailto:${presEmail}">${presEmail}</a> prior to the talk.<br>
<b>P.S.2</b>: GitHub Repo for lab talks - <a target="_blank" href="${talkGH}">${talkGH}</a><br>
<b>P.S.3</b>: Website for lab talks - <a target="_blank" href="${talkSite}">${talkSite}</a><br>

<b>N.B.</b>: This is an <b>automated</b> email.
`
var allMails = [];
for (var i = 1; i < data.length; i++) { // Skip row 1
  let canMail = data[i][1] == "Yes";
  if (canMail) // Skip people we are mailing separately / do not want on the mailing list
    allMails.push(data[i][2]);
}
var allMailsStr = allMails.join(","); // Convert to a string with comma-separated values
// console.log(allMailsStr);


// Some formatting (Font mainly) for the mail
function bodyFormatWrap(body) {
  // NOTE: Not all fonts are supported in Gmail or other email clients. Use common fonts.
  // Serif: georgia,garamond,serif
  return `<p style = "font-family:'Trebuchet MS',sans-serif;">${body}</p>`;
}

// Get the date difference in hours
function getDateDiffInHours(targetDate) {
  var currentDate = new Date();
  var targetDateObj = new Date(targetDate);
  var differenceInMs = targetDateObj.getTime() - currentDate.getTime();

	// Convert milliseconds to hours
  var diff = differenceInMs / (1000 * 60 * 60);

	return diff;
}

// Util function to send emails via MailApp | 1 or more email IDs
function sendMail(emailIDs, subject, emailBody) {
  // Format the email body
  emailBody = bodyFormatWrap(emailBody);

  // Send email
  if (pauseMails) {
    Logger.log("Emails are stopped for debugging.");
    Logger.log(subject);
    Logger.log(emailBody);
  } else {
    MailApp.sendEmail(
      emailIDs,
      subject,
      emailBody,
      {
        htmlBody: emailBody,
        noReply: false, // Will use own email ID
      }
    );
  }
  Logger.log(`Email sent to ${emailIDs} with subject: ${subject}`);
}


function sendAbstractMail(emailID, presenter, talkDateBod) {
  const subject = `SMLab Talks | Abstract Submission | ${presenter}`
  const emailBody = `Hello ${presenter},<br><br>

This is a reminder to share the title and abstract for your talk, scheduled on <b>${talkDateBod}</b>. You can fill the Google Form at this link - ${formLink} to submit this information. Please submit the form <b>at least 3 days before</b> your scheduled talk, so that invites can be sent out. The talk will be <b>cancelled</b> otherwise.<br><br>
${sign}
  `;

  // Send to emailID
  // See: https://developers.google.com/apps-script/reference/mail/mail-app#sendemailrecipient,-subject,-body,-options
  // Also see: https://developers.google.com/apps-script/reference/gmail/gmail-app#sendEmail(String,String,String,Object)
  sendMail(emailID, subject, emailBody);
}

function sendAbstractReminder(emailID, presenter, talkDateBod) {
  const subject = `SMLab Talks | Abstract Submission | ${presenter}`
  const emailBody = `Hello ${presenter},<br><br>

This is the <b>last</b> reminder to share the title and abstract for your talk, scheduled on <b>${talkDateBod}</b>. You can fill the Google Form at this link - ${formLink}, to submit this information. Please submit the form <b>at least 3 days before</b> your scheduled talk, so that invites can be sent out. The talk will be <b>cancelled</b> otherwise.<br><br>
${sign}
  `;

  // Send to emailID, same thread as AbstractMail (NOTE: MailApp handles it based on identical `subject`)
  // Also see: https://developers.google.com/apps-script/reference/gmail/gmail-thread#replyallbody,-options
  sendMail(emailID, subject, emailBody);
}

function sendTalkAnnouncement(presenter, talkDateBod, talkDateSub, title, abstract, extra, online) {
  const subject = `SMLab Talks | ${talkDateSub} | ${presenter} | ${title}`;
  const meetInfo = `The talk will be held <b>online</b>. The Google Meet information is as follows:<br>
${gMeetInfo}`
  const talkVenueMod = online ? "online" : `at ${talkVenue}`;
  const emailBody = `Hello everyone,<br><br>

${presenter} will be giving a talk <b>${talkVenueMod}</b> on <b>${talkDateBod}</b>. The details of the talk are as follows:<br><br>

<b>Title</b>: ${title}<br><br>

<b>Abstract</b>:<br>${abstract}<br><br>

${extra ? '<b>Extra</b>:<br>' + extra + '<br><br>' : ''}

${online ? meetInfo + '<br><br>' : ''}

You are all invited to attend the talk.<br><br>
${sign}<br><br>
  `;

  // Send to allMails
  sendMail(allMailsStr, subject, emailBody);

  // Add Calendar event if not paused
  // See: https://developers.google.com/apps-script/reference/calendar/calendar#createeventtitle,-starttime,-endtime,-options
  evtStart = new Date(talkDateBod);
  evtEnd = new Date(talkDateBod);
  evtEnd.setHours(evtEnd.getHours() + 1);
  Logger.log(debugCal ? "Debugging is ON. Using testCalID..." : "Debugging is OFF. Using sLabCalID...");
  var calendar = CalendarApp.getCalendarById(debugCal ? testCalID : sLabCalID); // NOTE: testCalID for testing
  calendar.createEvent(
    `SMLab Talks | ${presenter} | ${title}`,
    evtStart,
    evtEnd,
    {
      description: emailBody,
      location: online ? "Online" : talkVenue,
      sendInvites: false, // As we are sending the invites ourselves
    }
  );
  Logger.log(`Calendar event created for ${presenter} | ${title} | ${talkDateBod}`);
  // }
}

function sendTalkReminder1(presenter, talkDateBod, talkDateSub, title, online) {
  const subject = `SMLab Talks | ${talkDateSub} | ${presenter} | ${title}`
  const talkVenueMod = online ? "online" : `at ${talkVenue}`;
  const emailBody = `Hello everyone,<br><br>

This is a reminder for the lab talk by <b>${presenter}</b>, scheduled on <b>${talkDateBod}</b>. The talk will take place <b>${talkVenueMod}</b>. Other details are in the previous email.<br><br>
${sign}
  `;

  // Send to allMails, same thread as TalkAnnouncement
  sendMail(allMailsStr, subject, emailBody);
}

function sendTalkReminder2(presenter, talkDateBod, talkDateSub, title, online) {
  const subject = `SMLab Talks | ${talkDateSub} | ${presenter} | ${title}`
  const talkVenueMod = online ? "online" : `at ${talkVenue}`;
  const emailBody = `Hello everyone,<br><br>

This is the <b>last</b> reminder for the lab talk by <b>${presenter}</b>, scheduled on <b>${talkDateBod}</b>. The talk will take place <b>${talkVenueMod}</b>. Other details are in the previous email.<br><br>
${sign}
  `;

  // Send to allMails, same thread as TalkAnnouncement
  sendMail(allMailsStr, subject, emailBody);
}

function sendTalkCancelled(presenter, talkDateBod, talkDateSub) {
  const subject = `SMLab Talks | Talk Cancelled | ${talkDateSub} | ${presenter}`
  const emailBody = `Hello everyone,<br><br>

This is to inform you that the talk by ${presenter}, to be held on ${talkDateBod}, has been cancelled. Apologies for any inconvenience.<br><br>
${sign}
  `;

  // Send to allMails
  sendMail(allMailsStr, subject, emailBody);
}


/**
 * Main function to check for scheduled talks and send emails accordingly
 * TODO: Can be simplified
 */
function labTalkDaemon() {
  Logger.log("===================================== RUN START TIME: " + new Date() + " =====================================");
  // Loop over rows | Skip row 1
  for (let i = 1; i < data.length; i++) {
    let talkDate = data[i][0];
    talkDate = new Date(talkDate);
    talkDateSub = Utilities.formatDate(talkDate, 'Asia/Calcutta', 'MMMM dd, yyyy HH:mm:ss');
    talkDateBod = Utilities.formatDate(talkDate, 'Asia/Calcutta', 'EEEE, MMMM dd, yyyy HH:mm:ss Z');
    let emailID = data[i][2];
    let fullName = data[i][3];
    let firstName = fullName.split(",")[1].trim();

    let title = data[i][4];
    let abstract = data[i][5];
    let extra = data[i][6];
    let online = data[i][7];

    // Checkboxes
    let absMail = sheet.getRange(i+1, 9);
    let absRem = sheet.getRange(i+1, 10);
    let talkAnn = sheet.getRange(i+1, 11);
    let talkRem1 = sheet.getRange(i+1, 12);
    let talkRem2 = sheet.getRange(i+1, 13);
    let talkCan = sheet.getRange(i+1, 14);

    // Boolean for whether data exists
    dataExists = title && abstract && extra != errStr && online != errStr;
    Logger.log(`TalkDate: ${talkDate} | ${fullName} | Data Exists?: ${dataExists} => ${title} | ${abstract} | ${extra} | ${online}`);
    Logger.log(`TalkDateSub: ${talkDateSub} | TalkDateBod: ${talkDateBod}`);

    // Test all date comparisons | Keep for debugging
    Logger.log(`TalkDate: ${talkDate} | ${fullName} | absTrig: ${getDateDiffInHours(talkDate)} <= ${absTrig} <=> ${getDateDiffInHours(talkDate) <= absTrig}`);
    Logger.log(`TalkDate: ${talkDate} | ${fullName} | absRemTrig: ${getDateDiffInHours(talkDate)} <= ${absRemTrig} <=> ${getDateDiffInHours(talkDate) <= absRemTrig}`);
    Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkAnnTrig / Talk Cancel: ${getDateDiffInHours(talkDate)} <= ${talkAnnTrig} <=> ${getDateDiffInHours(talkDate) <= talkAnnTrig}`);
    Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem1Trig: ${getDateDiffInHours(talkDate)} <= ${talkRem1Trig} <=> ${getDateDiffInHours(talkDate) <= talkRem1Trig}`);
    Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem2Trig: ${getDateDiffInHours(talkDate)} <= ${talkRem2Trig} <=> ${getDateDiffInHours(talkDate) <= talkRem2Trig}`);

    // Test all mails | Keep for debugging with pauseMails=true
    // sendAbstractMail(emailID, firstName, talkDateBod);
    // sendAbstractReminder(emailID, firstName, talkDateBod);
    // sendTalkAnnouncement(firstName, talkDateBod, talkDateSub, title, abstract, extra, online);
    /* The above code appears to be a comment section in a JavaScript file. It includes a function call
    `sendTalkReminder1(firstName, talkDateBod, talkDateSub, title);` with placeholders for
    parameters such as `firstName`, `talkDateBod`, `talkDateSub`, and `title`. The purpose of this
    code is likely to remind or send a notification for a talk or presentation based on the provided
    parameters. */
    // sendTalkReminder1(firstName, talkDateBod, talkDateSub, title, online);
    // sendTalkReminder2(firstName, talkDateBod, talkDateSub, title, online);
    // sendTalkCancelled(firstName, talkDateBod, talkDateSub);
    // break;

    // Ignore negative time differences
    if (getDateDiffInHours(talkDate) <= 0) {
      continue;
    }

    // Reversed order of checks
    if (getDateDiffInHours(talkDate) <= talkRem2Trig) {
      // If 1 hour before the talk
      if (talkCan.getValue()) {
        // If talk is cancelled, log and do nothing
        // NOTE: This supercedes the check for existence of data
        Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem2Trig | Talk is cancelled`);
      } else {
        // If talk is not cancelled, check talkRem2
        if (!talkRem2.getValue()) {
          // If TalkReminder2 has not been sent, sendTalkReminder2
          sendTalkReminder2(firstName, talkDateBod, talkDateSub, title, online);
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem2Trig | Ran sendTalkReminder2`);
          // Check off TalkReminder2 in sheet
          talkRem2.setValue(true);
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem2Trig | Set talkRem2 to TRUE`);
        } else {
          // If TalkReminder2 has already been sent, log and do nothing
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem2Trig | TalkReminder2 already sent`);
        }
      }
    } else if (getDateDiffInHours(talkDate) <= talkRem1Trig) {
      // If the day of the talk
      if (talkCan.getValue()) {
        // If talk is cancelled, log and do nothing
        Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem1Trig | Talk is cancelled`);
      } else {
        // If talk is not cancelled, check talkRem1
        if (!talkRem1.getValue()) {
          // If TalkReminder1 has not been sent, sendTalkReminder1
          sendTalkReminder1(firstName, talkDateBod, talkDateSub, title, online);
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem1Trig | Ran sendTalkReminder1`);
          // Check off TalkReminder1 in sheet
          talkRem1.setValue(true);
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem1Trig | Set talkRem1 to TRUE`);
        } else {
          // If TalkReminder1 has already been sent, log and do nothing
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkRem1Trig | TalkReminder1 already sent`);
        }
      }
    } else if (getDateDiffInHours(talkDate) <= talkAnnTrig) {
      // If 2 days or less left from scheduled talk
      if (!dataExists && !talkCan.getValue()) {
        // If data does not exist, sendTalkCancelled
        // NOTE: This is automated cancelling, based on whether data exists within the stipulated time. DO NOT MANUALLY MARK THE CHECKBOX OFF TO CANCEL TALKS.
        // NOTE: i.e., talkCan should only be false if no data exists.
        sendTalkCancelled(firstName, talkDateBod, talkDateSub);
        Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkAnnTrig / talkCan | Ran sendTalkCancelled`);
        // Check off TalkCancelled in sheet
        talkCan.setValue(true);
        Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkAnnTrig / talkCan | Set talkCan to TRUE`);
      } else {
        if (dataExists && !talkAnn.getValue()) {
          // If data exists, and talk has not been announced, sendTalkAnnouncement
          sendTalkAnnouncement(firstName, talkDateBod, talkDateSub, title, abstract, extra, online);
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkAnnTrig / talkCan | Ran sendTalkAnnouncement`);
          // Check off TalkAnnouncement in sheet
          talkAnn.setValue(true);
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkAnnTrig / talkCan | Set talkAnn to TRUE`);
        } else {
          // if TalkAnnouncement has already been sent, log and do nothing
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | talkAnnTrig / talkCan | TalkAnnouncement already sent`);
        }
      }
		} else if (getDateDiffInHours(talkDate) <= absRemTrig) {
			// If 3 days or less left from scheduled talk
      if (!dataExists && !absRem.getValue() && absMail.getValue()) {
        // If data does not exist, AbsMail sent, but AbsRem not sent, sendAbstractReminder
        sendAbstractReminder(emailID, firstName, talkDateBod);
        Logger.log(`TalkDate: ${talkDate} | ${fullName} | absRemTrig | Ran sendAbstractReminder`);
        // Check off AbstractReminder in sheet
        absRem.setValue(true);
        Logger.log(`TalkDate: ${talkDate} | ${fullName} | absRemTrig | Set absRem to TRUE`);
      } else {
        // If data exists or AbsRem already sent, log and do nothing
        if (absRem.getValue()) {
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | absRemTrig | AbsRem already sent`);
        } else {
          Logger.log(`TalkDate: ${talkDate} | ${fullName} | absRemTrig | Talk data exists`);
        }
      }
    } else if (getDateDiffInHours(talkDate) <= absTrig) {
      // If 7 days or less left from scheduled talk
      if (!dataExists && !absMail.getValue()) {
        // If data does not exist & AbsMail not sent, sendAbstractMail
        sendAbstractMail(emailID, firstName, talkDateBod);
				Logger.log(`TalkDate: ${talkDate} | ${fullName} | absTrig | Ran sendAbstractMail`);
        // Check off AbstractMail in sheet
        absMail.setValue(true);
				Logger.log(`TalkDate: ${talkDate} | ${fullName} | absTrig | Set absMail to TRUE`);
      } else {
        // If data exists, log and do nothing
				Logger.log(`TalkDate: ${talkDate} | ${fullName} | absTrig | No talk data & AbsMail already sent`);
      }
    }
  }
  Logger.log("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx RUN END TIME: " + new Date() + " xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
}
