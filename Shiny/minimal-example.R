library(shiny)

# UI
ui <- fluidPage(
  titlePanel("My First Shiny App"),
  
  textInput("name", "What's your name?"),
  textOutput("greeting")
)

# Server
server <- function(input, output) {
  output$greeting <- renderText({
    paste("Hello,", input$name, "!")
  })
}

# Run the app
shinyApp(ui = ui, server = server)
