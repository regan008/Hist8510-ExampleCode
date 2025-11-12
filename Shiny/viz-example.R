library(shiny)

# Load the Philadelphia data
philly_data <- read.csv("data_philly.csv", stringsAsFactors = FALSE)

# Get unique types for the filter
venue_types <- c("All", sort(unique(unlist(strsplit(philly_data$type, ",")))))
venue_types <- venue_types[venue_types != "" & !is.na(venue_types)]

ui <- fluidPage(
  titlePanel("Philadelphia Historical Venues"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("venue_type", 
                  "Filter by Venue Type:", 
                  choices = venue_types,
                  selected = "All")
    ),
    
    mainPanel(
      plotOutput("typeChart")
    )
  )
)

server <- function(input, output) {
  
  # Filter data based on selected venue type
  filtered_data <- reactive({
    if (input$venue_type == "All") {
      philly_data
    } else {
      philly_data[grepl(input$venue_type, philly_data$type, fixed = TRUE), ]
    }
  })
  
  # Bar chart of venue types
  output$typeChart <- renderPlot({
    data <- filtered_data()
    
    # Split types and count
    all_types <- unlist(strsplit(data$type, ","))
    all_types <- trimws(all_types)
    all_types <- all_types[all_types != "" & !is.na(all_types)]
    
    type_counts <- table(all_types)
    type_counts <- sort(type_counts, decreasing = TRUE)
    
    if (length(type_counts) > 0) {
      barplot(type_counts, 
              col = 'steelblue', 
              border = 'white',
              main = 'Number of Venues by Type',
              xlab = 'Venue Type',
              ylab = 'Count',
              las = 2,  # Rotate labels
              cex.names = 0.8)
    }
  })
}

shinyApp(ui = ui, server = server)