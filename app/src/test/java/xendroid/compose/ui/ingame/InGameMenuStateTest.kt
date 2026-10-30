package xendroid.compose.ui.ingame

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class InGameMenuStateTest {
    private fun InGameMenuState.on(action: InGameAction) = select(actions().indexOf(action))

    @Test fun reopeningRestoresLastPageWithoutReopeningQuitConfirmation() {
        val session = InGameMenuState().show(pause = true).changePage(-1).on(InGameAction.QUIT).askToQuit()
        val reopened = session.hide().show(pause = false)
        assertEquals(InGamePage.SESSION, reopened.page)
        assertEquals(InGameAction.QUIT, reopened.action)
        assertFalse(reopened.confirmingQuit)
        assertFalse(reopened.pausedByMenu)
    }

    @Test fun navigationRetainsFocusPerPageAcrossOpenings() {
        val menu = InGameMenuState().show(pause = true).move(2)
        assertEquals(InGameAction.ANTIALIASING, menu.action)
        val system = menu.changePage(1)
        assertEquals(InGamePage.SYSTEM, system.page)
        assertEquals(InGameAction.FPS_LIMIT, system.action)
        val graphics = system.changePage(-1)
        assertEquals(2, graphics.selected)
        assertEquals(2, graphics.changePage(1).changePage(-1).selected)
        assertEquals(2, graphics.hide().show(pause = false).selected)
        assertFalse(graphics.hide().show(pause = false).pausedByMenu)
    }

    /** U01, round 2: nothing gets lost in the categories and groups, and nothing shows twice. */
    @Test fun everyOptionLivesOnExactlyOneCategoryAndGroup() {
        val placed = inGamePageActions.values.flatten()
        assertEquals(placed.size, placed.toSet().size)
        assertEquals(InGameAction.entries.toSet() - InGameAction.MORE_OPTIONS, placed.toSet())
        assertTrue(advancedActions.all { it in placed })
        inGamePageActions.forEach { (page, actions) -> actions.forEach { assertNotNull("$it on $page", groupOf(page, it)) } }
        assertTrue(inGamePageActions.getValue(InGamePage.GRAPHICS).containsAll(listOf(InGameAction.DISPLAY_MODE, InGameAction.ANTIALIASING,
            InGameAction.SHARPNESS, InGameAction.WINFG, InGameAction.DRIVER_INFO)))
        assertTrue(inGamePageActions.getValue(InGamePage.SYSTEM).containsAll(listOf(InGameAction.FPS_LIMIT, InGameAction.SUSTAINED_PERFORMANCE)))
        assertTrue(inGamePageActions.getValue(InGamePage.HUD).containsAll(listOf(InGameAction.PERFORMANCE_HUD, InGameAction.HUD_LAYOUT,
            InGameAction.HUD_METRICS, InGameAction.HUD_OPACITY, InGameAction.HUD_COLORS)))
        assertTrue(inGamePageActions.getValue(InGamePage.CONTROLS).containsAll(listOf(InGameAction.CONTROL_STYLE,
            InGameAction.EDIT_TOUCH_LAYOUT, InGameAction.PHONE_CONTROLLERS, InGameAction.GYRO_CAMERA)))
        assertTrue(inGamePageActions.getValue(InGamePage.SESSION).containsAll(listOf(InGameAction.PAUSE_ON_OPEN, InGameAction.AUTO_SAVE,
            InGameAction.UNDO_SESSION, InGameAction.MAKE_GLOBAL, InGameAction.SHARE_LOGS, InGameAction.QUIT)))
    }

    @Test fun moreOptionsOpenInPlaceAndPerCategory() {
        val menu = InGameMenuState().show(pause = true)
        val closed = menu.actions(InGamePage.SYSTEM)
        assertEquals(InGameAction.MORE_OPTIONS, closed.last())
        assertTrue(closed.none { it in advancedActions })
        val toggle = closed.indexOf(InGameAction.MORE_OPTIONS)
        val open = menu.changePage(1).select(toggle).toggleAdvanced()
        assertEquals(InGameAction.MORE_OPTIONS, open.action)                       // still on the toggle
        assertEquals(toggle, open.selected)
        assertEquals(closed, open.actions().take(closed.size))                      // common ones unchanged
        assertTrue(InGameAction.SUSTAINED_PERFORMANCE in open.actions() && InGameAction.BACKGROUND_POLICY in open.actions())
        assertEquals(3, open.advancedCount())                                      // sustained, ADPF hints, background
        assertEquals(closed.size + 3, open.count)
        // Other categories keep theirs closed; closing again hides them.
        assertTrue(open.actions(InGamePage.GRAPHICS).none { it in advancedActions })
        assertEquals(closed, open.toggleAdvanced().actions())
        // A selection inside the advanced list does not point past the end once closed.
        val deep = open.select(open.count - 1)
        assertEquals(toggle, deep.toggleAdvanced().selected)
    }

    @Test fun playerModeHidesDeveloperActionsFromNavigation() {
        val player = InGameMenuState(developer = false).show(pause = true)
        val graphics = player.actions(InGamePage.GRAPHICS)
        assertTrue(graphics.none { it in developerActions })
        // Frame generation is for players too.
        assertTrue(InGameAction.WINFG in graphics && InGameAction.LSFG in graphics)
        assertTrue(InGameAction.DISPLAY_MODE in graphics && InGameAction.SCALING_EFFECT in graphics && InGameAction.DRIVER_INFO in graphics)
        assertEquals(graphics.size, player.count)
        // Wrapping navigation stays inside the shown actions.
        assertEquals(graphics.last(), player.move(-1).action)
        // Open, the advanced list still has no developer option: dither, the colour filter, stretch
        // and LSFG's multiplier, target, import and cache.
        val graphicsOpen = player.toggleAdvanced()
        assertEquals(7, graphicsOpen.advancedCount())
        assertTrue(graphicsOpen.actions().none { it in developerActions })
        assertTrue(InGameAction.STRETCH in graphicsOpen.actions())
        assertTrue(InGameAction.PHONE_CONTROLLERS in player.actions(InGamePage.CONTROLS))
        // Performance's advanced options are all developer ones: no "More options" there in Player.
        assertTrue(InGameAction.MORE_OPTIONS !in player.actions(InGamePage.SYSTEM))
        assertTrue(InGameAction.MORE_OPTIONS in InGameMenuState().actions(InGamePage.SYSTEM))
    }

    @Test fun captureFrameOnlyShowsInTheDiagnosticBuild() {
        val normal = InGameMenuState().show(pause = true).actions(InGamePage.SESSION)
        assertTrue(InGameAction.CAPTURE_FRAME !in normal)
        assertTrue(InGameAction.SCREENSHOT in normal)
        // In both interface modes, beside the screenshot.
        for (developer in listOf(true, false)) {
            val diagnostic = InGameMenuState(developer = developer, frameTrace = true).show(pause = true)
                .actions(InGamePage.SESSION)
            assertEquals(diagnostic.indexOf(InGameAction.SCREENSHOT) + 1, diagnostic.indexOf(InGameAction.CAPTURE_FRAME))
            assertEquals(normal.size + 1, diagnostic.size)
        }
    }

    /** Round 2: a row of chips (the HUD's metrics) has a cursor of its own for ←→. */
    @Test fun aChipCursorMovesWithinItsRowAndStartsOverElsewhere() {
        val row = InGameMenuState().show(pause = true).copy(page = InGamePage.HUD).on(InGameAction.HUD_METRICS)
        assertEquals(InGameAction.HUD_METRICS, row.action)
        assertEquals(RowKind.MULTI, row.action!!.kind)
        val moved = row.moveChip(1, 8).moveChip(1, 8)
        assertEquals(2, moved.chip)
        assertEquals(7, moved.moveChip(10, 8).chip)
        assertEquals(0, moved.moveChip(-10, 8).chip)
        assertEquals(2, moved.select(moved.selected).chip)                         // the same row keeps it
        assertEquals(0, moved.move(1).chip)                                        // another row starts over
        assertEquals(0, moved.changePage(1).chip)
    }

    @Test fun everyRowHasAKindTheControllerCanWorkWith() {
        val adjustable = setOf(RowKind.CHOICE, RowKind.MULTI, RowKind.CYCLE, RowKind.TOGGLE, RowKind.SLIDER)
        // The options a player tunes the most are changed in place by ←→, not by a list of rows.
        listOf(InGameAction.DISPLAY_MODE, InGameAction.FPS_LIMIT, InGameAction.SCALING_EFFECT, InGameAction.VOLUME,
            InGameAction.HUD_LAYOUT, InGameAction.CONTROL_STYLE, InGameAction.PAUSE_ON_OPEN).forEach { assertTrue("$it", it.kind in adjustable) }
        assertEquals(RowKind.BUTTON, InGameAction.QUIT.kind)
        assertEquals(RowKind.INFO, InGameAction.DRIVER_INFO.kind)
    }

    @Test fun quitConfirmationCannotSwitchTabsAndCanBeCancelled() {
        val menu = InGameMenuState().show(pause = true).changePage(-1).on(InGameAction.QUIT).askToQuit()
        assertEquals(InGameAction.QUIT, menu.cancelQuit().action)
        assertEquals(InGamePage.SESSION, menu.changePage(1).page)
        assertEquals(1, menu.move(1).selected)
        assertTrue(menu.hide().show(pause = false).open)
        assertFalse(menu.hide().show(pause = false).confirmingQuit)
    }
}
